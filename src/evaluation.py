"""
evaluation.py — 모델 로딩, 배치 생성, 채점·집계 라이브러리 (계획서 3-4)

직접 실행하지 않는다. 명령행은 src/run.py (추론·평가), 학습 중 검증과 학습 후 평가는 train.py,
배포 모델 평가는 release.py 가 이 파일의 함수를 쓴다.

- 평가셋 하나 = 평가 폴더(config.EVAL_DIR) 아래 폴더 하나: pairs.json (질문, SQL) + sql.json (그 SQL 들의 메타데이터).
  sql.json 은 손으로 만들지 않고 sql_gen 출력에서 pairs.json 에 쓰인 SQL 만 뽑아 만든다(write_sql_meta).
  폴더 하나로 완결되므로 새 평가셋은 폴더만 추가하면 된다. 메타데이터 없는 임의의 쌍 목록도 evaluate_pairs 로 채점한다.
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
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import torch

from src import config
from src.data.dataset import BOS_ID, EOS_ID, PAD_ID, SEP_ID, tokenize_pairs
from src.models.model import TextToSQLModel
from src.tokenizer import bpe



# ---------------------------------------------------------------------------
# 생성
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, examples: list[dict], device, amp_dtype,
                   batch_size: int = 512, max_new_tokens: int = 64) -> list[list[int]]:
    """각 예시의 프롬프트(<bos> question <sep>) 뒤를 greedy 로 생성해 SQL 토큰 ID 를 반환한다 (<eos> 제외).

    오른쪽 패딩 + causal attention 구조라 길이가 다른 프롬프트를 한 배치에 넣을 수 없다
    (왼쪽 패딩은 attention mask 가 없어 패딩을 보게 된다). 그래서 프롬프트 길이(= sql_start)가
    같은 것끼리 묶어 배치 생성한다. 반환 순서는 입력 순서와 같다.
    """
    was_training = model.training
    model.eval()
    groups: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex["sql_start"], []).append(i)

    outputs: list[list[int]] = [[] for _ in examples]
    for plen, idxs in groups.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            ids = torch.tensor([examples[i]["ids"][:plen] for i in chunk], dtype=torch.long, device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            for _ in range(min(max_new_tokens, model.max_seq_len - plen)):
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    next_id = model(ids)[:, -1].argmax(-1)
                next_id = torch.where(done, torch.full_like(next_id, PAD_ID), next_id)
                ids = torch.cat([ids, next_id[:, None]], dim=1)
                done |= next_id == EOS_ID
                if done.all():
                    break
            for i, row in zip(chunk, ids[:, plen:].tolist()):
                gen = row[:row.index(EOS_ID)] if EOS_ID in row else row
                outputs[i] = [t for t in gen if t != PAD_ID]
    model.train(was_training)
    return outputs


def predict_questions(model, questions: list[str], tok: bpe.Tokenizer, device, amp_dtype,
                      batch_size: int = 512) -> list[str | None]:
    """정답 없는 질문 목록을 배치로 SQL 로 바꾼다. 질문은 학습 데이터처럼 소문자로 바꿔 넣는다.
    프롬프트가 모델 최대 길이를 넘거나 디코딩할 수 없는 출력이면 None."""
    examples, idx = [], []
    for i, q in enumerate(questions):
        ids = [BOS_ID] + tok.encode(q.strip().lower()) + [SEP_ID]
        if len(ids) < model.max_seq_len:
            examples.append({"ids": ids, "sql_start": len(ids)})
            idx.append(i)
    preds: list[str | None] = [None] * len(questions)
    for i, out in zip(idx, generate_batch(model, examples, device, amp_dtype, batch_size)):
        preds[i] = safe_decode(out, tok)
    return preds


def safe_decode(ids: list[int], tok: bpe.Tokenizer) -> str | None:
    """생성 결과가 깨진 UTF-8 이거나 토크나이저에 없는 ID 면 None (항상 오답 처리).
    특수 토큰은 "<sep>" 같은 문자열로 복원되므로 None 은 아니지만 정답 SQL 과 일치할 수 없어 역시 오답이 된다."""
    try:
        return tok.decode(ids)
    except (UnicodeDecodeError, KeyError):
        return None


def split_example(ex: dict, tok: bpe.Tokenizer) -> tuple[str, str]:
    """토큰화된 예시에서 (질문, 정답 SQL) 문자열을 복원한다."""
    plen = ex["sql_start"]
    return tok.decode(ex["ids"][1:plen - 1]), tok.decode(ex["ids"][plen:-1])


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


# ---------------------------------------------------------------------------
# 채점 보조
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 평가셋 폴더
# ---------------------------------------------------------------------------

def list_sets(eval_dir: str | Path = config.EVAL_DIR) -> list[str]:
    """pairs.json 이 있는 하위 폴더 이름 (정렬)."""
    d = Path(eval_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if (p / config.PAIRS_FILE).is_file())


def build_sql_meta(pairs_path: str | Path, sql_path: str | Path) -> list[dict]:
    """sql_path(sql_gen 출력) 중 pairs_path 에 나오는 SQL 의 메타데이터를 원래 순서대로 반환한다."""
    with open(pairs_path, encoding="utf-8") as f:
        used = {p["sql"] for p in json.load(f)}
    with open(sql_path, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if r["sql"] in used]
    missing = used - {r["sql"] for r in rows}
    if missing:
        raise ValueError(f"{sql_path} 에 없는 SQL {len(missing)}개 (예: {sorted(missing)[0]!r})")
    return rows


def write_sql_meta(name: str, sql_path: str | Path, eval_dir: str | Path = config.EVAL_DIR) -> Path:
    d = Path(eval_dir) / name
    rows = build_sql_meta(d / config.PAIRS_FILE, sql_path)
    out = d / config.EVAL_SQL_FILE
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out


def load_eval_set(name: str, eval_dir: str | Path = config.EVAL_DIR) -> tuple[list[dict], dict[str, dict]]:
    """(쌍 목록, SQL → 메타데이터). 쌍의 SQL 이 sql.json 에 하나라도 없으면 오류 — sql.json 이 낡았다는 뜻이다."""
    d = Path(eval_dir) / name
    with open(d / config.PAIRS_FILE, encoding="utf-8") as f:
        pairs = json.load(f)
    if not pairs:
        raise ValueError(f"{d / config.PAIRS_FILE} 에 평가 쌍이 없음")
    with open(d / config.EVAL_SQL_FILE, encoding="utf-8") as f:
        meta = {r["sql"]: r for r in json.load(f)}
    missing = {p["sql"] for p in pairs} - meta.keys()
    if missing:
        raise ValueError(f"{d / config.EVAL_SQL_FILE} 에 없는 SQL {len(missing)}개 — "
                         f"python -m src.evaluation --build-sql {name} --sql <sql 파일> 로 다시 만든다")
    return pairs, meta


# ---------------------------------------------------------------------------
# 평가셋 단위 평가
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 모델 불러오기
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str | Path, device) -> tuple[TextToSQLModel, dict]:
    """학습 체크포인트(best.pt)와 배포 모델(model.pt) 모두 읽는다. 둘 다 텐서와 기본 자료형만 담고 있어
    weights_only=True 로 읽는다 (pickle 로 임의 코드가 실행되지 않게)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TextToSQLModel(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def default_out_dir(ckpt_path: str | Path, release_eval_dir: str | Path = config.RELEASE_EVAL_DIR) -> Path:
    """평가 결과(eval_<set>.json) 기본 위치. 출력은 항상 runs/ 아래(커밋 안 함)에 둔다:
    runs/ 안의 학습 체크포인트면 그 실행 폴더, 그 밖(배포 모델 등)이면 release_eval_dir."""
    runs_root = config.RUNS_DIR.parent.resolve()
    if runs_root in Path(ckpt_path).resolve().parents:
        return Path(ckpt_path).parent
    return Path(release_eval_dir)


def describe_checkpoint(ckpt: dict) -> str:
    """출력용 요약: 학습 체크포인트면 epoch·검증 EM, 배포 모델(model.pt)이면 단계."""
    if "epoch" in ckpt:
        return f"epoch {ckpt['epoch']}, val EM {ckpt.get('val_em', 0):.4f}"
    return f"배포 모델, stage {ckpt.get('stage')}"


def tokenizer_path_for(ckpt_path: str | Path, ckpt: dict) -> Path:
    """체크포인트를 학습할 때 쓴 토크나이저. 체크포인트 폴더에 사본(train.py 가 저장)이 있으면 그것을 쓴다 —
    코퍼스나 seed 목록이 바뀌어 config.TOKENIZER_PATH 가 다시 만들어져도 이전 체크포인트를 그대로 평가하기 위해서다.
    예전 실행 폴더에는 병합 규칙만 있는 bpe_merges.json 이 있고, Tokenizer.load 가 그 형식도 읽는다."""
    folder = Path(ckpt_path).parent
    for name in ("tokenizer.json", "bpe_merges.json"):
        if (folder / name).exists():
            return folder / name
    return Path(ckpt.get("tokenizer_path") or ckpt.get("merges_path") or config.TOKENIZER_PATH)
