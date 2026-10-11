"""
scoring.py — 생성 결과 채점·집계 (계획서 3-4)

직접 실행하지 않는다. 학습 중 검증과 학습 후 평가는 train.py, 배포 모델 평가는 release.py, 명령행 평가는
run.py --evaluation 이 이 파일의 함수를 쓴다. 평가셋 폴더는 eval_sets.py, 생성은 src/inference/generate.py.

- evaluate_set 은 평가셋 폴더 하나를, evaluate_pairs 는 메타데이터 없는 임의의 (질문, 정답 SQL) 목록도 채점한다.
- 주 지표 Exact Match(EM): greedy 생성 SQL 이 정답 SQL 문자열과 완전히 같은 비율.
- 보조 지표 실행 정확도: shop.db 에서 두 SQL 의 실행 결과가 같은 비율. 행 수가 적은 DB 라 틀린 SQL 도
  우연히 같은 결과를 낼 수 있으므로(예: 0행) 보조로만 본다.
- 계층(분포 내 / 미등장 개체 / 미등장 ID / 미등장 주문 ID)·WHERE 컬럼별로 나눠 집계하고, EM 에는 95%
  신뢰구간(Wilson)을 붙인다. 틀린 사례는 어느 부분(테이블·WHERE 컬럼·SELECT·리터럴)이 틀렸는지 분류한다.
- 다중 정답 EM(multi): 질문만으로 정답이 하나로 정해지지 않는 문항은 다른 답도 정답으로 본다. 기존 EM 과
  별도로 보고한다.
- 규칙 일치 EM(rule_consistent): 정답 라벨이 라벨 규칙(계획서 4-5)과 다른 문항을 뺀 EM. 평가셋은 정제하지 않아
  이런 문항에서는 규칙대로 답한 모델이 오답 처리된다. 판정은 단계 규칙의 label_mismatch(질문, 정답)로 하며
  모델 출력은 보지 않는다. 기존 EM 과 함께 보고한다.
- 단계마다 다른 규칙(SQL 분해, 오답 부분 분류, 다중 정답 판정)은 src/data/stage<N>/sql_rules.py 에 있고
  load_rules(stage) 로 고른다. 이 파일에는 단계 공통 부분만 둔다.
- 제약 디코더는 아직 없으므로 지금 수치는 계획서의 "제약 디코더 미적용" 수치(게이트 판정용)다.
"""

from __future__ import annotations

import importlib
import math
import sqlite3
from collections import Counter
from pathlib import Path

from src import config
from src.data.dataset import tokenize_pairs
from src.evaluation.eval_sets import load_eval_set
from src.inference.generate import generate_batch, safe_decode, split_example
from src.tokenizer import bpe


def load_rules(stage: int = config.STAGE):
    """단계별 SQL 규칙 모듈 (src.data.stage<N>.sql_rules)."""
    return importlib.import_module(f"src.data.stage{stage}.sql_rules")


def execute(con: sqlite3.Connection, sql: str | None) -> Counter | None:
    """실행 결과를 순서 무관 multiset 으로. 실행 오류면 None."""
    if sql is None:
        return None
    try:
        return Counter(con.execute(sql).fetchall())
    except sqlite3.Error:
        return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 95% Wilson 신뢰구간 (계획서 3-4: 게이트 판정 시 점추정과 함께 기록)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(records: list[dict]) -> dict:
    n = len(records)
    k = sum(r["em"] for r in records)
    lo, hi = wilson_ci(k, n)
    km = sum(r["em_multi"] for r in records)
    return {"n": n, "em": k / n if n else 0.0, "em_ci95": [lo, hi],
            "em_multi": km / n if n else 0.0, "em_multi_ci95": list(wilson_ci(km, n)),
            "exec_acc": sum(r["exec_match"] for r in records) / n if n else 0.0}


def greedy_exact_match(model, examples: list[dict], tok: bpe.Tokenizer, device, amp_dtype
                       ) -> tuple[float, list[dict]]:
    """EM 과 실패 사례만 빠르게 낸다 (학습 중 epoch 마다 검증용)."""
    preds = generate_batch(model, examples, device, amp_dtype)
    n_correct, failures = 0, []
    for ex, pred_ids in zip(examples, preds):
        question, gold = split_example(ex, tok)
        pred = safe_decode(pred_ids, tok)
        if pred == gold:
            n_correct += 1
        else:
            failures.append({"question": question, "gold": gold, "pred": pred})
    return n_correct / len(examples), failures


def evaluate_set(model, name: str, tok: bpe.Tokenizer, device, amp_dtype,
                 eval_dir: str | Path = config.EVAL_DIR, db_path: str | Path = config.DB_PATH,
                 rules=None, batch_size: int = 512) -> dict:
    """평가셋 하나(<eval_dir>/<name>/)를 평가한다 (evaluate_pairs 참고)."""
    pairs, meta = load_eval_set(name, eval_dir)
    return evaluate_pairs(model, pairs, tok, device, amp_dtype, name, meta, db_path, rules, batch_size)


def evaluate_pairs(model, pairs: list[dict], tok: bpe.Tokenizer, device, amp_dtype, name: str,
                   meta: dict[str, dict] | None = None, db_path: str | Path = config.DB_PATH,
                   rules=None, batch_size: int = 512) -> dict:
    """(질문, 정답 SQL) 쌍을 평가해 요약(전체·계층별·WHERE 컬럼별·오류 유형)과 틀린 사례 전체를 반환한다.
    meta: SQL → 메타데이터(table, where_col, holdout_tier). 없으면(임의의 쌍 파일) 계층은 name 하나로 묶고
    테이블·WHERE 컬럼은 단계 규칙으로 SQL 에서 뽑는다.
    rules: 단계별 SQL 규칙 모듈 (기본: config.STAGE 의 load_rules())."""
    rules = rules or load_rules()
    label_mismatch = getattr(rules, "label_mismatch", lambda q, s: False)
    meta = meta or {}
    examples = tokenize_pairs(pairs, tok)
    preds = generate_batch(model, examples, device, amp_dtype, batch_size)

    con = sqlite3.connect(db_path)
    records = []
    for p, ex, pred_ids in zip(pairs, examples, preds):
        pred = safe_decode(pred_ids, tok)
        m = meta.get(p["sql"])
        if m is None:
            g = rules.parse_sql(p["sql"]) or {}
            m = {"table": g.get("table"), "where_col": g.get("where_col")}
        em = pred == p["sql"]
        gold_res = execute(con, p["sql"])
        records.append({
            "question": p["question"], "gold": p["sql"], "pred": pred, "em": em,
            "em_multi": rules.multi_match(p["question"], p["sql"], pred),
            "select_ambiguous": rules.select_ambiguous(p["question"], p["sql"]),
            "label_mismatch": label_mismatch(p["question"], p["sql"]),
            "exec_match": em or (gold_res is not None and execute(con, pred) == gold_res),
            "tier": m.get("holdout_tier", name), "table": m.get("table"), "where_col": m.get("where_col"),
            "error_parts": [] if em else rules.error_parts(p["sql"], pred),
        })
    con.close()

    def group_by(key) -> dict:
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(key(r)), []).append(r)
        return {g: summarize(rs) for g, rs in sorted(groups.items())}

    failures = [r for r in records if not r["em"]]
    error_counts = Counter(part for r in failures for part in r["error_parts"])
    return {
        "set": name, "multi_note": rules.MULTI_NOTE, "overall": summarize(records),
        "rule_consistent": summarize([r for r in records if not r["label_mismatch"]]),
        "n_label_mismatch": sum(r["label_mismatch"] for r in records),
        "by_tier": group_by(lambda r: r["tier"]),
        "by_where_col": group_by(lambda r: f"{r['table']}.{r['where_col']}"),
        "error_parts": dict(error_counts.most_common()),
        "n_select_ambiguous": sum(r["select_ambiguous"] for r in records),
        "failures": failures,
    }


def print_report(result: dict) -> None:
    def line(label: str, s: dict) -> str:
        lo, hi = s["em_ci95"]
        return (f"  {label:<28}{s['n']:>6}  EM {s['em']:6.1%} [{lo:5.1%}, {hi:5.1%}]  "
                f"multi {s['em_multi']:6.1%}  exec {s['exec_acc']:6.1%}")
    print(f"== {result['set']}")
    print(line("전체", result["overall"]))
    if result.get("n_label_mismatch"):
        print(line("규칙 일치 문항만", result["rule_consistent"]) + f"  (라벨 불일치 {result['n_label_mismatch']}개 제외)")
    if len(result["by_tier"]) > 1:
        print(" 계층별")
        for g, s in result["by_tier"].items():
            print(line(g, s))
    print(" WHERE 컬럼별")
    for g, s in result["by_where_col"].items():
        print(line(g, s))
    print(f" 틀린 부분 (오답 {len(result['failures'])}개, 한 오답에 여러 부분 가능): {result['error_parts']}")
    print(f" 다중 정답 인정 문항 {result['n_select_ambiguous']}개 ({result['multi_note']})")
