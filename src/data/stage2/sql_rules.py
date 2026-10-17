"""
sql_rules.py — 2단계 채점 규칙 (조건 0~3개, AND / OR, IN, 비교·BETWEEN·LIKE, ORDER BY, LIMIT)

evaluation/scoring.py 가 load_rules(2) 로 불러 쓴다. 1단계 sql_rules 와 같은 이름을 제공하고(그 docstring 참고), 2단계에서 더한
것은 아래 둘이다.

    same_sql(gold, pred) -> bool                EM 판정. IN 값 순서와 AND·OR 조건 순서를 무시한다(라벨 규칙 L8)
    breakdowns(pair, meta) -> dict[str, str]    리포트를 나눠 볼 기준 → 값 (BREAKDOWNS 의 이름, 해당 없으면 뺀다)

- EM: 학습 쌍의 IN 값은 질문에 나온 순서이고(L8) 값 순서만 다른 IN 은 결과가 같으므로, 예측과 정답을 정규형(sql_ast.canonical)
  으로 바꿔 비교한다. 리터럴 표기(price 의 소수 둘째 자리 등)와 공백은 정규형과 같아야 한다 — "price < 50" 은 오답이다.
- 다중 정답 EM: 1단계와 같은 뜻이다. 정답 SELECT 가 * 또는 식별 컬럼이고 질문에 어느 쪽 단서도 없으면 다른 쪽도 정답으로 본다.
  존재 확인(L1)은 제외한다. 판정은 질문과 정답만 보고 하며, 정한 뒤에는 라벨 규칙에 맞춰 고치지 않는다.
- 규칙 일치 EM: 2단계 라벨 규칙 판정(clean_ambiguous.drop_reason)이 어긋났다고 보는 문항을 뺀다.
"""

from __future__ import annotations

from src.data.stage1.sql_rules import ID_COL, ID_CUE, STAR_CUE
from src.data.stage2 import sql_ast
from src.data.stage2.clean_ambiguous import _strip_conditions, drop_reason
from src.data.stage2.sql_gen import GenConfig

STAGE = 2
MULTI_NOTE = "SELECT 모호 문항: * ↔ 식별 컬럼도 정답 인정 (IN 값·조건 순서는 EM 에서도 무시)"
SQL_PARTS = ("table", "select", "where_col", "op", "literal", "conn", "order", "limit")
# 리포트에 나눠 볼 기준 (이름 → 표시 이름). 값이 없는 문항(예: 페르소나가 없는 조합기 문장)은 그 기준에서 빠진다
BREAKDOWNS = {"split": "분할", "writer": "작성자", "persona": "페르소나", "struct_combo": "구조 조합",
              "holdout_shape": "미등장 값 모양", "eval_slot": "eval 표현 자리", "table": "테이블",
              "conds": "조건", "order": "정렬·LIMIT"}


def _parse(sql: str | None) -> sql_ast.Query | None:
    """2단계 문법으로 분해한다. 문법 밖이거나 그 테이블에 없는 컬럼(예측의 "items.membership")이면 None."""
    try:
        q = sql_ast.parse(sql or "")
        q.render()  # 조건 컬럼이 테이블에 없으면 리터럴 표기를 정할 수 없어 KeyError
        sql_ast.canonical(q)
    except (ValueError, KeyError):
        return None
    return q


def parse_sql(sql: str | None) -> dict | None:
    """2단계 문법으로 분해한다. 맞지 않으면 None. where_col 은 조건 컬럼을 스키마 순서로 이은 것("-" 은 조건 없음)."""
    q = _parse(sql)
    if q is None:
        return None
    c = sql_ast.canonical(q)
    return {"table": c.table, "select": c.select, "where_col": "+".join(x.col for x in c.conds) or "-", "query": c}


def _norm(sql: str | None) -> str | None:
    """정규형 문자열. 표기가 정규형 렌더링과 다르면(리터럴 형식·공백) None."""
    q = _parse(sql)
    if q is None or q.render() != sql:
        return None
    return sql_ast.canonical(q).render()


def same_sql(gold: str, pred: str | None) -> bool:
    if pred == gold:
        return True
    p = _norm(pred)
    return p is not None and p == _norm(gold)


def error_parts(gold: str, pred: str | None) -> list[str]:
    """정답과 다른 부분 목록. 2단계 문법으로 분해되지 않으면 ['malformed']."""
    g, p = _parse(gold), _parse(pred)
    if g is None or p is None:
        return ["malformed"]
    g, p = sql_ast.canonical(g), sql_ast.canonical(p)
    gc, pc = {c.col: c for c in g.conds}, {c.col: c for c in p.conds}
    shared = gc.keys() & pc.keys()
    parts = {
        "table": g.table != p.table,
        "select": g.select != p.select,
        "where_col": gc.keys() != pc.keys(),
        "op": any(gc[k].op != pc[k].op for k in shared),
        "literal": any(gc[k].op == pc[k].op and (gc[k].val, gc[k].val2) != (pc[k].val, pc[k].val2) for k in shared),
        "conn": len(g.conds) >= 2 and len(p.conds) >= 2 and g.conn != p.conn,
        "order": g.order != p.order,
        "limit": g.limit != p.limit,
    }
    return [k for k in SQL_PARTS if parts[k]]


def select_ambiguous(question: str, gold: str) -> bool:
    """정답 SELECT 가 * 또는 식별 컬럼이고, 질문에 둘 중 어느 쪽을 원하는지 단서가 없으면 True.
    존재 확인(SELECT 가 = 조건 컬럼)은 표기 규칙이 정해져 있으므로 제외한다. WHERE 조건을 말하는 표현("named amy",
    orders 질문의 "item id 3155", ID 값 앞의 "id")은 단서로 보지 않는다."""
    g = _parse(gold)
    if g is None:
        return False
    if g.select not in ("*", ID_COL[g.table]) or g.select in {c.col for c in g.conds if c.op == "="}:
        return False
    q = _strip_conditions(question.lower(), g)
    return not STAR_CUE.search(q) and not ID_CUE[g.table].search(q)


def multi_match(question: str, gold: str, pred: str | None) -> bool:
    """다중 정답 EM: EM 이거나, 모호한 문항에서 SELECT 만 * ↔ 식별 컬럼으로 다른 경우."""
    if same_sql(gold, pred):
        return True
    if not select_ambiguous(question, gold) or _norm(pred) is None:
        return False
    g, p = sql_ast.canonical(_parse(gold)), sql_ast.canonical(_parse(pred))
    if p.select not in ("*", ID_COL[g.table]):
        return False
    swapped = sql_ast.Query(p.table, g.select, p.conds, p.conn, p.order, p.limit)
    return swapped.render() == g.render()


def label_mismatch(question: str, gold: str) -> bool:
    """정답 라벨이 2단계 라벨 규칙(clean_ambiguous.drop_reason)과 다르면 True. 평가 오답을 보고 규칙을 고치지 않는다."""
    return drop_reason({"question": question, "sql": gold}) is not None


def _combo_label(combo: int) -> str:
    names = {"n": lambda v: f"조건 {v}개", "limit": lambda v: "LIMIT", "order": lambda v: "ORDER BY",
             "op": lambda v: str(v), "table": lambda v: str(v)}
    return " × ".join(names[k](v) for k, v in GenConfig().struct_holdout[combo])


def breakdowns(pair: dict, meta: dict | None) -> dict[str, str]:
    """리포트를 나눠 볼 기준별 값. 쌍(pair)에 있는 출처 정보(split, writer, persona, eval_slot), SQL 메타데이터(struct_combo,
    holdout_shape), 정답 SQL 구조(테이블, 조건 수·연결어·IN·LIKE, 정렬·LIMIT)에서 뽑는다."""
    meta = meta or {}
    out = {k: str(pair[k]) for k in ("split", "writer", "persona") if pair.get(k) is not None}
    if pair.get("eval_slot"):
        out["eval_slot"] = pair["eval_slot"].split(":")[0]
    if meta.get("struct_combo") is not None:
        out["struct_combo"] = _combo_label(int(meta["struct_combo"]))
    if meta.get("holdout_shape"):
        out["holdout_shape"] = meta["holdout_shape"]
    g = _parse(pair["sql"])
    if g is not None:
        ops = {c.op for c in g.conds}
        conds = f"{len(g.conds)}개" + (f" {g.conn}" if g.conn else "")
        conds += "".join(f" +{op}" for op in ("IN", "LIKE") if op in ops)
        out.update(table=g.table, conds=conds,
                   order="-" if g.order is None else "ORDER BY" + ("" if g.limit is None else
                                                                   " + LIMIT 1" if g.limit == 1 else " + LIMIT n"))
    return {k: out[k] for k in BREAKDOWNS if k in out}
