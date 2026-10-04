"""
evaluation.py — 학습된 체크포인트를 평가셋으로 평가한다 (계획서 3-4)

저장소 루트에서 실행한다.

    python -m src.evaluation --ckpt runs/stage1_ffn683/best.pt               # 분포 내 + 미등장 값 평가셋
    python -m src.evaluation --ckpt runs/stage1_ffn683/best.pt --sets holdout

- 주 지표 Exact Match(EM): greedy 생성 SQL 이 정답 SQL 문자열과 완전히 같은 비율.
- 보조 지표 실행 정확도: shop.db 에서 두 SQL 의 실행 결과가 같은 비율. 행 수가 적은 DB 라 틀린 SQL 도
  우연히 같은 결과를 낼 수 있으므로(예: 0행) 보조로만 본다.
- 계층(분포 내 / 미등장 개체 / 미등장 ID / 미등장 주문 ID)·WHERE 컬럼별로 나눠 집계하고, EM 에는 95%
  신뢰구간(Wilson)을 붙인다. 틀린 사례는 어느 부분(테이블·WHERE 컬럼·SELECT·리터럴)이 틀렸는지 분류한다.
- 결과: 체크포인트 폴더에 eval_<set>.json (요약 + 틀린 사례 전체).
- 다중 정답 EM(multi): 개체를 묻지만 식별 컬럼(name/item_name/order_id)과 행 전체(*) 중 무엇을 원하는지
  질문에 단서가 없는 문항은 둘 다 정답으로 본다 (select_ambiguous). 기존 EM 과 별도로 보고한다.
- 제약 디코더는 아직 없으므로 지금 수치는 계획서의 "제약 디코더 미적용" 수치(게이트 판정용)다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import torch

from src import config
from src.data.dataset import EOS_ID, PAD_ID, load_pairs, tokenize_pairs
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

# 평가셋 이름 → (질문 쌍 glob, SQL 메타데이터 파일). 폴더 이름은 절대 pilot* 이면 안 된다 (학습 데이터로 잡힘).
EVAL_SETS = {
    "indist": ("eval_indist*/eval_pairs.json", "sql_eval_indist.json"),
    "holdout": ("eval_holdout*/eval_pairs.json", "sql_eval_holdout.json"),
    # 2026-10-04 이전 미등장 값 평가셋 (Qwen3-14B 생성, 표현 다양성 낮음). 이전 실행과 비교용
    "holdout_qwen": ("eval_qwen_holdout*/eval_pairs.json", "sql_eval_holdout.json"),
}
DB_PATH = Path(config.DATA_DIR) / "shop.db"

SQL_PATTERN = re.compile(r"^SELECT (\*|\w+) FROM (\w+) WHERE (\w+) = (.+)$")
SQL_PARTS = ("table", "where_col", "select", "literal")


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


def safe_decode(ids: list[int], id_to_bytes) -> str | None:
    """생성 결과가 깨진 UTF-8 이거나 merges 에 없는 ID 면 None (항상 오답 처리).
    특수 토큰은 "<sep>" 같은 문자열로 복원되므로 None 은 아니지만 정답 SQL 과 일치할 수 없어 역시 오답이 된다."""
    try:
        return bpe.decode(ids, id_to_bytes)
    except (UnicodeDecodeError, KeyError):
        return None


def split_example(ex: dict, id_to_bytes) -> tuple[str, str]:
    """토큰화된 예시에서 (질문, 정답 SQL) 문자열을 복원한다."""
    plen = ex["sql_start"]
    return bpe.decode(ex["ids"][1:plen - 1], id_to_bytes), bpe.decode(ex["ids"][plen:-1], id_to_bytes)


def greedy_exact_match(model, examples: list[dict], id_to_bytes, device, amp_dtype
                       ) -> tuple[float, list[dict]]:
    """EM 과 실패 사례만 빠르게 낸다 (학습 중 epoch 마다 검증용)."""
    preds = generate_batch(model, examples, device, amp_dtype)
    n_correct, failures = 0, []
    for ex, pred_ids in zip(examples, preds):
        question, gold = split_example(ex, id_to_bytes)
        pred = safe_decode(pred_ids, id_to_bytes)
        if pred == gold:
            n_correct += 1
        else:
            failures.append({"question": question, "gold": gold, "pred": pred})
    return n_correct / len(examples), failures


# ---------------------------------------------------------------------------
# 채점 보조
# ---------------------------------------------------------------------------

def parse_sql(sql: str | None) -> dict | None:
    """1단계 문법(SELECT x FROM t WHERE c = v)으로 분해한다. 맞지 않으면 None."""
    m = SQL_PATTERN.match(sql or "")
    if not m:
        return None
    return dict(zip(("select", "table", "where_col", "literal"), m.groups()))


def error_parts(gold: str, pred: str | None) -> list[str]:
    """정답과 다른 부분 목록. 1단계 문법으로 분해되지 않으면 ['malformed']."""
    g, p = parse_sql(gold), parse_sql(pred)
    if g is None or p is None:
        return ["malformed"]
    return [k for k in SQL_PARTS if g[k] != p[k]]


# 다중 정답 EM: "which item costs 14.46?" 처럼 개체를 묻지만 이름(식별 컬럼)만 원하는지 행 전체(*)를
# 원하는지 질문에 드러나지 않는 문항은 둘 중 무엇을 SELECT 해도 정답으로 본다. 판정은 질문과 정답 SQL 만
# 보고 하며 모델 출력은 보지 않는다. 기존 EM 은 그대로 두고 별도 지표(em_multi)로 보고한다.
ID_COL = {"customers": "name", "items": "item_name", "orders": "order_id"}
STAR_CUE = re.compile(
    r"\b(all|everything|every|full|whole|complete|entire|details?|info|information|records?|entry|entries"
    r"|rows?|data|profile|lowdown|rundown|about|overview|summary|anything|columns?|fields?"
    r"|on file|scoop|deal with|going on)\b")
ID_CUE = {
    "customers": re.compile(r"\b(names?|named|called)\b"),
    "items": re.compile(r"\b(names?|named|called)\b"),
    "orders": re.compile(r"\b(order[ _]?ids?|order numbers?|ids?|numbers?)\b"),
}


def select_ambiguous(question: str, gold: str) -> bool:
    """정답 SELECT 가 * 또는 식별 컬럼이고, 질문에 둘 중 어느 쪽을 원하는지 단서가 없으면 True.
    SELECT 와 WHERE 컬럼이 같은 존재 확인형(규칙 10)은 표기 규칙이 정해져 있으므로 제외한다."""
    g = parse_sql(gold)
    if g is None:
        return False
    id_col = ID_COL[g["table"]]
    if g["select"] not in ("*", id_col) or g["select"] == g["where_col"]:
        return False
    q = question.lower()
    # WHERE 조건을 말하는 표현("item id 3155", "named vivian")은 SELECT 단서가 아니다
    if g["table"] == "orders":
        q = re.sub(r"\b(item|customer)[ _]?ids?\b", " ", q)
    q = re.sub(rf"\b(named|called)\s+{re.escape(g['literal'].strip(chr(39)))}\b", " ", q)
    return not STAR_CUE.search(q) and not ID_CUE[g["table"]].search(q)


def multi_match(question: str, gold: str, pred: str | None) -> bool:
    """다중 정답 EM: 완전 일치이거나, 모호한 문항에서 SELECT 만 * ↔ 식별 컬럼으로 다른 경우."""
    if pred == gold:
        return True
    if not select_ambiguous(question, gold):
        return False
    g, p = parse_sql(gold), parse_sql(pred)
    return (p is not None and p["select"] in ("*", ID_COL[g["table"]])
            and all(g[k] == p[k] for k in ("table", "where_col", "literal")))


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
# 평가셋 단위 평가
# ---------------------------------------------------------------------------

def evaluate_set(model, name: str, merges, id_to_bytes, device, amp_dtype,
                 data_dir: str | Path = config.DATA_DIR) -> dict:
    """평가셋 하나를 평가해 요약(전체·계층별·WHERE 컬럼별·오류 유형)과 틀린 사례 전체를 반환한다."""
    pair_glob, sql_file = EVAL_SETS[name]
    pairs = load_pairs(data_dir, pair_glob)
    if not pairs:
        raise FileNotFoundError(f"{data_dir}/{pair_glob} 에 평가 쌍이 없음")
    meta = {r["sql"]: r for r in json.load(open(Path(data_dir) / sql_file, encoding="utf-8"))}
    examples = tokenize_pairs(pairs, merges)
    preds = generate_batch(model, examples, device, amp_dtype)

    con = sqlite3.connect(DB_PATH)
    records = []
    for p, ex, pred_ids in zip(pairs, examples, preds):
        pred = safe_decode(pred_ids, id_to_bytes)
        m = meta.get(p["sql"], {})
        em = pred == p["sql"]
        gold_res = execute(con, p["sql"])
        records.append({
            "question": p["question"], "gold": p["sql"], "pred": pred, "em": em,
            "em_multi": multi_match(p["question"], p["sql"], pred),
            "select_ambiguous": select_ambiguous(p["question"], p["sql"]),
            "exec_match": em or (gold_res is not None and execute(con, pred) == gold_res),
            "tier": m.get("holdout_tier", name), "table": m.get("table"), "where_col": m.get("where_col"),
            "error_parts": [] if em else error_parts(p["sql"], pred),
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
        "set": name, "overall": summarize(records),
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
    print(f" SELECT 모호 문항 {result['n_select_ambiguous']}개 (multi = 이 문항에서 * ↔ 식별 컬럼도 정답 인정)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str | Path, device) -> tuple[TextToSQLModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TextToSQLModel(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def merges_path_for(ckpt_path: str | Path, ckpt: dict) -> Path:
    """체크포인트를 학습할 때 쓴 BPE merges. 실행 폴더에 사본(train.py 가 저장)이 있으면 그것을 쓴다 —
    코퍼스가 바뀌어 루트의 bpe_merges.json 이 다시 만들어져도 이전 체크포인트를 그대로 평가하기 위해서다."""
    local = Path(ckpt_path).parent / "bpe_merges.json"
    return local if local.exists() else Path(ckpt.get("merges_path", config.BPE_MERGES_PATH))


def main() -> int:
    ap = argparse.ArgumentParser(description="체크포인트 평가")
    ap.add_argument("--ckpt", required=True, help="train.py 가 저장한 best.pt")
    ap.add_argument("--sets", nargs="+", default=list(EVAL_SETS), choices=list(EVAL_SETS))
    ap.add_argument("--out-dir", default=None, help="eval_<set>.json 저장 위치 (기본: 체크포인트 폴더)")
    ap.add_argument("--show", type=int, default=10, help="평가셋마다 출력할 틀린 사례 수")
    ap.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None
    model, ckpt = load_model(args.ckpt, device)
    merges = bpe.load_merges(merges_path_for(args.ckpt, ckpt))
    id_to_bytes = bpe.id_to_bytes_from_merges(merges)
    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint {args.ckpt} (epoch {ckpt.get('epoch')}, val EM {ckpt.get('val_em', 0):.4f}, "
          f"params {model.num_params():,})")
    for name in args.sets:
        result = evaluate_set(model, name, merges, id_to_bytes, device, amp_dtype)
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
