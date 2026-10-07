"""학습 쌍 중 라벨 규칙(계획서 4-5)과 어긋나는 쌍을 제거한다 (라벨 통일).

질문만으로 정답 SQL 이 하나로 정해지도록, 규칙과 다른 라벨이 붙은 쌍을 지운다. 규칙의 근거와 예시는
계획서 4-5 에 있고, 여기서는 그중 코드로 판정하는 세 가지를 적용한다.
  L2 SELECT `*` / 식별 컬럼 — 단서가 없으면(`evaluation.select_ambiguous`) `select_label` 순서로 정한다
  L3 ID / 이름 — 상품·고객 자신의 ID(item_id, customer_id)는 질문이 ID 를 요구할 때만 SELECT 한다
  L4 테이블 — customers/items 로도 성립하는 orders SQL 은 질문에 주문 단서가 있을 때만 쓴다

평가셋은 이전 실행과 비교할 수 있도록 건드리지 않는다.

    python -m src.data.clean_ambiguous            # dry-run: 제거 대상 수와 예시만 출력
    python -m src.data.clean_ambiguous --apply    # pilot_merged 에서 제거, 제거 목록을 data_raw/ 에 누적 저장
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.evaluation import ID_COL, parse_sql, select_ambiguous

TRAIN_FILE = Path("data/pilot_merged/pilot_train_pairs.json")
REMOVED_FILE = Path("data_raw/clean_ambiguous_removed.json")

REASONS = {"select": "L2 SELECT 모호", "id": "L3 ID/이름", "table": "L4 테이블 모호"}

PLURAL = re.compile(r"\b(items|products|orders|customers)\b")
WH_ID = re.compile(r"\b(which|who)\b")
ROW_VERB = re.compile(
    r"\b(show|pull(ing)? up|look(ing)? ?up|look like|have on|what'?s up|what'?s in|what is in|story"
    r"|tell me about)\b")
OWN_ID = {"items": "item_id", "customers": "customer_id"}
ID_CUE = re.compile(r"\b(ids?|identifiers?|numbers?)\b|#")
ORDER_CUE = re.compile(
    r"\b(orders?|ordered|ordering|purchases?|purchased|bought|buys?|transactions?|shipments?|shipped"
    r"|deliver\w*|pending|cancel\w*)\b")


def select_label(question: str, g: dict) -> str:
    """L2: SELECT 모호 문항의 정답 SELECT. 위에서부터 먼저 맞는 것을 쓴다."""
    id_col = ID_COL[g["table"]]
    q = question.lower()
    if g["where_col"] == id_col:   # name = 'henry' 에 SELECT name 은 존재 확인형이 된다
        return "*"
    if WH_ID.search(q):
        return id_col
    if PLURAL.search(q):
        return "*"
    if ROW_VERB.search(q):
        return "*"
    return id_col


def drop_reason(pair: dict) -> str | None:
    """규칙과 어긋나면 사유 키("select" / "id" / "table"), 아니면 None."""
    g = parse_sql(pair["sql"])
    if g is None:
        return None
    q = pair["question"].lower()
    if select_ambiguous(pair["question"], pair["sql"]) and g["select"] != select_label(q, g):
        return "select"
    # SELECT == WHERE 컬럼인 존재 확인형은 L1 이 따로 정한다
    if (g["select"] == OWN_ID.get(g["table"]) and g["select"] != g["where_col"]
            and not ID_CUE.search(q)):
        return "id"
    if (g["table"] == "orders" and g["where_col"] in ("customer_id", "item_id")
            and g["select"] in ("*", g["where_col"]) and not ORDER_CUE.search(q)):
        return "table"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="라벨 규칙과 어긋나는 학습 쌍 제거")
    ap.add_argument("--apply", action="store_true", help="실제로 학습 파일을 고쳐 쓴다")
    args = ap.parse_args()

    pairs = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
    keep, removed = [], []
    for p in pairs:
        reason = drop_reason(p)
        if reason:
            removed.append({**p, "reason": reason})
        else:
            keep.append(p)

    counts = Counter(r["reason"] for r in removed)
    detail = ", ".join(f"{label} {counts[key]}" for key, label in REASONS.items())
    print(f"학습 쌍 {len(pairs)}개 중 제거 {len(removed)}개 ({detail}) -> {len(keep)}개")
    for key in REASONS:
        for r in [r for r in removed if r["reason"] == key][:5]:
            print(f"  [{key}] {r['question']}  ->  {r['sql']}")

    if args.apply and removed:
        # 원본 형식 유지: indent 2, CRLF, 끝 줄바꿈 없음
        TRAIN_FILE.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8", newline="\r\n")
        # 규칙을 추가해 다시 돌려도 이전 제거 목록이 남도록 누적한다
        prev = json.loads(REMOVED_FILE.read_text(encoding="utf-8")) if REMOVED_FILE.exists() else []
        REMOVED_FILE.parent.mkdir(parents=True, exist_ok=True)
        REMOVED_FILE.write_text(json.dumps(prev + removed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"저장: {TRAIN_FILE}, 제거 목록(누적 {len(prev) + len(removed)}개): {REMOVED_FILE}")
    elif not args.apply:
        print("dry-run: --apply 를 주면 학습 파일을 고쳐 쓴다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
