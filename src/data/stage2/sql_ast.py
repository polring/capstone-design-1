"""
2단계 SQL 구조(AST), 정규형 렌더링, 역파싱, 구조 규칙 검사.

SQL 문자열과 (나중에) 질문은 모두 이 구조에서 만든다. 같은 의미의 SQL이 두 가지 문자열로
나오지 않도록 정규형을 하나로 고정한다.

정규형
    SELECT {sel} FROM {t} [WHERE c1 {AND|OR} c2 ...] [ORDER BY col ASC|DESC] [LIMIT n]
    - 같은 컬럼의 값 나열은 col IN (v1, v2[, v3]) 조건 하나다. AND / OR 는 서로 다른 컬럼의 조건만 잇고, 섞지 않는다.
      OR 는 ID·이름(유일 키) 조건을 잇지 않는다.
    - AND·OR 조건은 스키마 컬럼 순서, IN 값은 값 순서로 정렬한다. 이것은 SQL 의 정체(분할·겹침 검사·IN 순서 무시 비교)용이고,
      학습 쌍의 IN 값은 질문에 나온 순서다 (라벨 규칙 L8, compose.question_order_sql).
    - 정렬 방향은 ASC 도 항상 쓴다.
    - 범위는 col BETWEEN lo AND hi 로만 쓴다 (같은 컬럼에 >= AND <= 금지).
    - LIKE 패턴은 'x%' / '%x' / '%x%' 세 가지만 쓴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.data.stage2 import schema

MAX_CONDS = 3
MAX_LIMIT = 10
LIKE_RE = re.compile(r"^(%?)([a-z]+)(%?)$")


@dataclass(frozen=True)
class Cond:
    col: str
    op: str
    val: object
    val2: object = None  # BETWEEN 상한

    def render(self, table: str) -> str:
        if self.op == "IN":
            return f"{self.col} IN (" + ", ".join(schema.fmt_literal(table, self.col, v) for v in self.val) + ")"
        lit = schema.fmt_literal(table, self.col, self.val)
        if self.op == "BETWEEN":
            return f"{self.col} BETWEEN {lit} AND {schema.fmt_literal(table, self.col, self.val2)}"
        return f"{self.col} {self.op} {lit}"


@dataclass(frozen=True)
class Query:
    table: str
    select: str
    conds: tuple[Cond, ...] = ()
    conn: str | None = None              # 조건이 2개 이상일 때 "AND" / "OR" (IN 은 조건 하나)
    order: tuple[str, str] | None = None  # (컬럼, "ASC" | "DESC")
    limit: int | None = None
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def render(self) -> str:
        parts = [f"SELECT {self.select} FROM {self.table}"]
        if self.conds:
            parts.append("WHERE " + f" {self.conn or 'AND'} ".join(c.render(self.table) for c in self.conds))
        if self.order:
            parts.append(f"ORDER BY {self.order[0]} {self.order[1]}")
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        return " ".join(parts)

    def cond_cols(self) -> set[str]:
        return {c.col for c in self.conds}

    def ops(self) -> list[str]:
        return [c.op for c in self.conds]


def _value_key(v):
    return (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))


def canonical(q: Query) -> Query:
    """조건 순서를 정규형으로 맞춘다 (AND·OR: 스키마 컬럼 순서, IN 값: 값 순서). 조건이 1개 이하면 conn 은 None."""
    conds = tuple(sorted((Cond(c.col, c.op, tuple(sorted(c.val, key=_value_key))) if c.op == "IN" else c
                          for c in q.conds), key=lambda c: schema.col_index(q.table, c.col)))
    conn = q.conn if len(conds) >= 2 else None
    return Query(q.table, q.select, conds, conn, q.order, q.limit, q.meta)


# ---------------------------------------------------------------------------
# 역파싱 (정규형 SQL 문자열 -> Query)
# ---------------------------------------------------------------------------

_LIT = r"'[^']*'|-?\d+(?:\.\d+)?"
_COND_RE = re.compile(
    rf"(?P<col>\w+) (?:(?P<op>!=|<=|>=|=|<|>|LIKE) (?P<val>{_LIT})"
    rf"|BETWEEN (?P<lo>{_LIT}) AND (?P<hi>{_LIT})"
    rf"|IN \((?P<vals>(?:{_LIT})(?:, (?:{_LIT}))*)\))")
_SQL_RE = re.compile(
    r"^SELECT (?P<sel>\*|\w+) FROM (?P<table>\w+)"
    r"(?: WHERE (?P<where>.+?))?"
    r"(?: ORDER BY (?P<ocol>\w+) (?P<odir>ASC|DESC))?"
    r"(?: LIMIT (?P<limit>\d+))?$")


def _parse_lit(s: str):
    if s.startswith("'"):
        return s[1:-1]
    return float(s) if "." in s else int(s)


def parse(sql: str) -> Query:
    """정규형 문법의 SQL 문자열을 Query 로 되돌린다. 문법 밖이면 ValueError."""
    m = _SQL_RE.match(sql.strip())
    if not m:
        raise ValueError(f"문법 밖 SQL: {sql!r}")
    conds: list[Cond] = []
    conn = None
    where = m["where"]
    pos = 0
    while where is not None and pos < len(where):
        cm = _COND_RE.match(where, pos)
        if not cm:
            raise ValueError(f"조건 해석 실패: {where[pos:]!r}")
        if cm["op"]:
            conds.append(Cond(cm["col"], cm["op"], _parse_lit(cm["val"])))
        elif cm["vals"]:
            conds.append(Cond(cm["col"], "IN", tuple(_parse_lit(v) for v in re.findall(_LIT, cm["vals"]))))
        else:
            conds.append(Cond(cm["col"], "BETWEEN", _parse_lit(cm["lo"]), _parse_lit(cm["hi"])))
        pos = cm.end()
        if pos < len(where):
            jm = re.compile(r" (AND|OR) ").match(where, pos)
            if not jm:
                raise ValueError(f"연결어 해석 실패: {where[pos:]!r}")
            if conn and conn != jm[1]:
                raise ValueError("AND 와 OR 를 섞은 SQL 은 2단계 문법 밖")
            conn = jm[1]
            pos = jm.end()
    order = (m["ocol"], m["odir"]) if m["ocol"] else None
    limit = int(m["limit"]) if m["limit"] else None
    return Query(m["table"], m["sel"], tuple(conds), conn, order, limit)


# ---------------------------------------------------------------------------
# 구조 규칙
# ---------------------------------------------------------------------------

def violations(q: Query) -> list[str]:
    """2단계 문법·라벨 규칙 위반 목록 (빈 목록이면 통과)."""
    v: list[str] = []
    t = q.table
    if t not in schema.TABLE_COLUMNS:
        return [f"없는 테이블 {t}"]
    cols = schema.TABLE_COLUMNS[t]
    if q.select != "*" and q.select not in cols:
        v.append(f"없는 SELECT 컬럼 {q.select}")

    # 조건 하나하나
    for c in q.conds:
        if c.col not in cols:
            v.append(f"없는 조건 컬럼 {c.col}")
            continue
        k = schema.kind(t, c.col)
        if c.op not in schema.ops(t, c.col):
            v.append(f"{c.col}({k}) 에 {c.op} 불가")
        if c.op == "IN" and not (isinstance(c.val, tuple) and 2 <= len(c.val) <= schema.IN_MAX
                                 and len(set(c.val)) == len(c.val)):
            v.append(f"IN 값은 서로 다른 2~{schema.IN_MAX}개: {c.val!r}")
        if c.op == "BETWEEN" and not (c.val2 is not None and c.val < c.val2):
            v.append(f"BETWEEN 하한 < 상한 아님: {c.val}, {c.val2}")
        if c.op != "BETWEEN" and c.val2 is not None:
            v.append("BETWEEN 이 아닌데 val2 있음")
        if c.op == "LIKE":
            lm = LIKE_RE.match(str(c.val))
            if not lm or not (lm[1] or lm[3]):
                v.append(f"LIKE 패턴은 'x%' / '%x' / '%x%' 만: {c.val!r}")

    # 조건 개수·연결어
    n = len(q.conds)
    if n > MAX_CONDS:
        v.append(f"조건 {n}개 > {MAX_CONDS}")
    if n >= 2:
        if q.conn not in ("AND", "OR"):
            v.append("조건 2개 이상인데 연결어 없음")
        elif len(q.cond_cols()) != n:
            v.append(f"{q.conn} 조건은 컬럼이 서로 달라야 함 (같은 컬럼 값 나열은 IN, 범위는 BETWEEN)")
        if q.conn == "OR" and (bad := [c.col for c in q.conds if schema.kind(t, c.col) == "id"
                                       or ((t, c.col) in schema.UNIQUE_KEYS and c.op in ("=", "IN"))]):
            v.append(f"OR 는 ID·이름 조건을 잇지 않음: {bad}")
        if q.conn == "OR" and q.limit is not None:
            v.append("OR 에는 LIMIT 을 붙이지 않음 (\"the 3 cheapest … in kitchen or under $10\" 은 범위가 모호)")
        if q.conn == "OR" and any(c.op == "!=" for c in q.conds):
            v.append("OR 의 항에는 != 를 쓰지 않음 (\"in chicago or not silver\" 는 부자연스러움)")
    elif q.conn is not None:
        v.append("조건 1개 이하인데 연결어 있음")

    # 유일 키 = / IN 조건은 단독으로만 (결과가 그 행들로 정해져 다른 조건·정렬이 무의미)
    has_unique_eq = any(c.op in ("=", "IN") and (t, c.col) in schema.UNIQUE_KEYS for c in q.conds)
    if has_unique_eq and (n >= 2 or q.order):
        v.append("유일 키 = / IN 조건에 다른 조건·ORDER BY 를 붙일 수 없음")

    # SELECT 가 = 조건 컬럼이면 단일 조건 존재 확인(L1)에만 쓴다. 같은 값을 되풀이하는 결과라 정렬도 붙이지 않는다.
    # IN 컬럼 SELECT 는 나열한 값을 되풀이하는 결과라 만들지 않는다
    in_cols = {c.col for c in q.conds if c.op == "IN"}
    eq_cols = {c.col for c in q.conds if c.op == "="} | in_cols
    if q.select in eq_cols and (n >= 2 or q.select in in_cols):
        v.append("조건이 여러 개이거나 IN 일 때 SELECT 를 그 조건 컬럼으로 둘 수 없음")
    if q.select in eq_cols and q.order:
        v.append("SELECT 가 = 조건 컬럼(존재 확인)이면 ORDER BY 를 붙이지 않음")

    # ORDER BY / LIMIT
    if q.order:
        ocol, odir = q.order
        if ocol not in cols or not schema.orderable(t, ocol):
            v.append(f"정렬 불가 컬럼 {ocol}")
        if odir not in ("ASC", "DESC"):
            v.append(f"정렬 방향 {odir}")
        if ocol in eq_cols:
            v.append("= 조건으로 고정된 컬럼으로 정렬")
    if q.limit is not None:
        if not q.order:
            v.append("LIMIT 은 ORDER BY 가 있을 때만")
        if not 1 <= q.limit <= MAX_LIMIT:
            v.append(f"LIMIT {q.limit} 범위 밖 (1~{MAX_LIMIT})")
    if n == 0 and not q.order:
        v.append("WHERE 없는 SQL 은 ORDER BY 가 있을 때만")

    if not v and canonical(q).render() != q.render():
        v.append("정규형 순서가 아님")
    return v


# ---------------------------------------------------------------------------
# 직렬화 · 특징
# ---------------------------------------------------------------------------

def to_dict(q: Query) -> dict:
    return {
        "sql": q.render(), "table": q.table, "select": q.select,
        "conds": [{"col": c.col, "op": c.op, "val": c.val, **({"val2": c.val2} if c.op == "BETWEEN" else {})}
                  for c in q.conds],
        "conn": q.conn,
        "order": {"col": q.order[0], "dir": q.order[1]} if q.order else None,
        "limit": q.limit,
        "skeleton": skeleton(q),
        **q.meta,
    }


def from_dict(d: dict) -> Query:
    conds = tuple(Cond(c["col"], c["op"], tuple(c["val"]) if c["op"] == "IN" else c["val"], c.get("val2"))
                  for c in d["conds"])
    order = (d["order"]["col"], d["order"]["dir"]) if d["order"] else None
    return Query(d["table"], d["select"], conds, d["conn"], order, d["limit"])


def limit_kind(limit: int | None) -> str:
    return "-" if limit is None else ("1" if limit == 1 else "n")


def skeleton(q: Query) -> str:
    """리터럴을 뺀 구조 서명. 예: items|sel=ident|AND|category=,price<|ord=price:ASC|lim=n"""
    sk = schema.select_kind(q.table, q.select, q.cond_cols())
    conds = ",".join(f"{c.col}{c.op}" for c in q.conds) or "-"
    order = f"{q.order[0]}:{q.order[1]}" if q.order else "-"
    return f"{q.table}|sel={sk}|{q.conn or '-'}|{conds}|ord={order}|lim={limit_kind(q.limit)}"


def features(q: Query) -> set[tuple[str, object]]:
    """구조 홀드아웃 조합을 판정할 원자 특징 집합."""
    f: set[tuple[str, object]] = {("table", q.table), ("n", len(q.conds)), ("conn", q.conn),
                                  ("order", q.order is not None), ("limit", q.limit is not None)}
    f |= {("op", c.op) for c in q.conds}
    return f
