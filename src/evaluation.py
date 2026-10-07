"""
evaluation.py — 학습된 체크포인트를 평가셋으로 평가한다 (계획서 3-4)

저장소 루트에서 실행한다.

    python -m src.evaluation --ckpt runs/stage1/ffn683_swap40/best.pt               # 평가셋 전부
    python -m src.evaluation --ckpt runs/stage1/ffn683_swap40/best.pt --sets holdout
    python -m src.evaluation --list                                                  # 평가셋 목록
    python -m src.evaluation --build-sql indist --sql data/stage1/sql/eval_indist.json   # 평가셋의 sql.json 생성

- 평가셋 하나 = config.EVAL_DIR 아래 폴더 하나: pairs.json (질문, SQL) + sql.json (그 SQL 들의 메타데이터).
  sql.json 은 손으로 만들지 않고 sql_gen 출력에서 pairs.json 에 쓰인 SQL 만 뽑아 만든다(--build-sql).
  폴더 하나로 완결되므로 새 평가셋은 폴더만 추가하면 된다.

- 주 지표 Exact Match(EM): greedy 생성 SQL 이 정답 SQL 문자열과 완전히 같은 비율.
- 보조 지표 실행 정확도: shop.db 에서 두 SQL 의 실행 결과가 같은 비율. 행 수가 적은 DB 라 틀린 SQL 도
  우연히 같은 결과를 낼 수 있으므로(예: 0행) 보조로만 본다.
- 계층(분포 내 / 미등장 개체 / 미등장 ID / 미등장 주문 ID)·WHERE 컬럼별로 나눠 집계하고, EM 에는 95%
  신뢰구간(Wilson)을 붙인다. 틀린 사례는 어느 부분(테이블·WHERE 컬럼·SELECT·리터럴)이 틀렸는지 분류한다.
- 결과: 체크포인트 폴더에 eval_<set>.json (요약 + 틀린 사례 전체).
- 다중 정답 EM(multi): 질문만으로 정답이 하나로 정해지지 않는 문항은 다른 답도 정답으로 본다. 기존 EM 과
  별도로 보고한다.
- 단계마다 다른 규칙(SQL 분해, 오답 부분 분류, 다중 정답 판정)은 src/data/stage<N>/sql_rules.py 에 있고,
  체크포인트에 저장된 stage(없으면 config.STAGE, --stage 로 변경)로 고른다. 이 파일에는 단계 공통 부분만 둔다.
- 제약 디코더는 아직 없으므로 지금 수치는 계획서의 "제약 디코더 미적용" 수치(게이트 판정용)다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import torch

from src import config
from src.data.dataset import EOS_ID, PAD_ID, tokenize_pairs
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
                 rules=None) -> dict:
    """평가셋 하나(<eval_dir>/<name>/)를 평가해 요약(전체·계층별·WHERE 컬럼별·오류 유형)과 틀린 사례 전체를 반환한다.
    rules: 단계별 SQL 규칙 모듈 (기본: config.STAGE 의 load_rules())."""
    rules = rules or load_rules()
    pairs, meta = load_eval_set(name, eval_dir)
    examples = tokenize_pairs(pairs, tok)
    preds = generate_batch(model, examples, device, amp_dtype)

    con = sqlite3.connect(db_path)
    records = []
    for p, ex, pred_ids in zip(pairs, examples, preds):
        pred = safe_decode(pred_ids, tok)
        m = meta[p["sql"]]
        em = pred == p["sql"]
        gold_res = execute(con, p["sql"])
        records.append({
            "question": p["question"], "gold": p["sql"], "pred": pred, "em": em,
            "em_multi": rules.multi_match(p["question"], p["sql"], pred),
            "select_ambiguous": rules.select_ambiguous(p["question"], p["sql"]),
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
# main
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str | Path, device) -> tuple[TextToSQLModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TextToSQLModel(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def tokenizer_path_for(ckpt_path: str | Path, ckpt: dict) -> Path:
    """체크포인트를 학습할 때 쓴 토크나이저. 체크포인트 폴더에 사본(train.py 가 저장)이 있으면 그것을 쓴다 —
    코퍼스나 seed 목록이 바뀌어 config.TOKENIZER_PATH 가 다시 만들어져도 이전 체크포인트를 그대로 평가하기 위해서다.
    예전 실행 폴더에는 병합 규칙만 있는 bpe_merges.json 이 있고, Tokenizer.load 가 그 형식도 읽는다."""
    folder = Path(ckpt_path).parent
    for name in ("tokenizer.json", "bpe_merges.json"):
        if (folder / name).exists():
            return folder / name
    return Path(ckpt.get("tokenizer_path") or ckpt.get("merges_path") or config.TOKENIZER_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description="체크포인트 평가 / 평가셋 관리")
    ap.add_argument("--ckpt", help="평가할 체크포인트 (train.py 가 저장한 best.pt)")
    ap.add_argument("--sets", nargs="+", default=None, help="평가셋 이름 (--eval-dir 아래 폴더, 기본: 전부)")
    ap.add_argument("--eval-dir", default=str(config.EVAL_DIR), help=f"평가셋 상위 폴더 (기본 {config.EVAL_DIR})")
    ap.add_argument("--out-dir", default=None, help="eval_<set>.json 저장 위치 (기본: 체크포인트 폴더)")
    ap.add_argument("--show", type=int, default=10, help="평가셋마다 출력할 틀린 사례 수")
    ap.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기")
    ap.add_argument("--stage", type=int, default=None,
                    help="채점 규칙 단계 (src/data/stage<N>/sql_rules.py, 기본: 체크포인트의 stage, 없으면 config.STAGE)")
    ap.add_argument("--list", action="store_true", help="평가셋 목록과 쌍·SQL 수를 출력하고 끝낸다")
    ap.add_argument("--build-sql", metavar="NAME", help="평가셋 NAME 의 sql.json 을 --sql 파일에서 만들고 끝낸다")
    ap.add_argument("--sql", help="--build-sql 용 sql_gen 출력 파일 (예: data/stage1/sql/eval_indist.json)")
    args = ap.parse_args()

    if args.list:
        for name in list_sets(args.eval_dir):
            pairs, meta = load_eval_set(name, args.eval_dir)
            print(f"{name:<16} 쌍 {len(pairs):>6}  SQL {len(meta):>5}")
        return 0
    if args.build_sql:
        if not args.sql:
            ap.error("--build-sql 에는 --sql 이 필요하다")
        print(f"{write_sql_meta(args.build_sql, args.sql, args.eval_dir)} 생성")
        return 0
    if not args.ckpt:
        ap.error("--ckpt 가 필요하다 (평가셋 관리만 하려면 --list / --build-sql)")
    available = list_sets(args.eval_dir)
    sets = args.sets or available
    unknown = sorted(set(sets) - set(available))
    if unknown:
        ap.error(f"{args.eval_dir} 에 없는 평가셋: {unknown} (있는 것: {available})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None
    model, ckpt = load_model(args.ckpt, device)
    tok = bpe.Tokenizer.load(tokenizer_path_for(args.ckpt, ckpt))
    stage = args.stage or ckpt.get("stage", config.STAGE)
    rules = load_rules(stage)
    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint {args.ckpt} (epoch {ckpt.get('epoch')}, val EM {ckpt.get('val_em', 0):.4f}, "
          f"params {model.num_params():,}, 채점 규칙 {stage}단계)")
    for name in sets:
        result = evaluate_set(model, name, tok, device, amp_dtype, eval_dir=args.eval_dir, rules=rules)
        print_report(result)
        for f in result["failures"][:args.show]:
            print(f"    Q: {f['question']}\n       gold: {f['gold']}\n       pred: {f['pred']}  "
                  f"({'+'.join(f['error_parts'])})")
        with open(out_dir / f"eval_{name}.json", "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
        print(f"  → {out_dir / f'eval_{name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
