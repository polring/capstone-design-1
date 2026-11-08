# 역할: 3단계 SQL의 구조 표현, 문자열 렌더링·역파싱과 정규화를 구현한다.
# 사용처: 3단계 SQL 생성·질문 조합·데이터 검사와 정답 비교의 공통 표현으로 사용한다.

"""
3단계 SQL 구조(AST), 정규형 렌더링, 역파싱, 구조 규칙 검사.

일반 조회·날짜 조건과 집계·GROUP BY·HAVING·깊이 1 스칼라 서브쿼리를 함께 처리한다.

SQL 문자열과 (나중에) 질문은 모두 이 구조에서 만든다. 같은 의미의 SQL이 두 가지 문자열로
나오지 않도록 정규형을 하나로 고정한다.

정규형
    SELECT {sel} FROM {t} [WHERE c1 {AND|OR} c2 ...] [ORDER BY col ASC|DESC] [LIMIT n]
    - 같은 컬럼의 값 나열은 col IN (v1, v2[, v3]) 조건 하나다. AND / OR 는 서로 다른 컬럼의 조건만 잇고, 섞지 않는다.
      OR 역시 ID·이름·부정 조건과 결합할 수 있다.
    - AND·OR 조건은 스키마 컬럼 순서, IN 값은 값 순서로 정렬한다. 이것은 SQL 의 정체(분할·겹침 검사·IN 순서 무시 비교)용이고,
      학습 쌍의 IN 값은 질문에 나온 순서다 (라벨 규칙 L8, compose.question_order_sql).
    - 정렬 방향은 ASC 도 항상 쓴다.
    - 새 정렬 SQL은 자기 기본키 ASC를 보조 키로 쓴다. 역파싱한 과거 SQL의 정렬은 보존한다.
    - 범위는 col BETWEEN lo AND hi 로만 쓴다 (같은 컬럼에 >= AND <= 금지).
    - LIKE 패턴은 'x%' / '%x' / '%x%' 세 가지만 쓴다.
"""

from __future__ import annotations

import re
from datetime import date
from dataclasses import dataclass, field, replace

from src.data.stage3 import schema

MAX_CONDS = 3
MAX_LIMIT = 10
LIKE_RE = re.compile(r"^(%?)([a-z]+)(%?)$")


def like_mode(pattern: str) -> tuple[str, str]:
    """LIKE 패턴을 접두·접미·포함 방향과 검색 조각으로 나눈다."""
    frag = pattern.strip("%")
    if pattern.startswith("%") and pattern.endswith("%"):
        return "contains", frag
    return ("suffix", frag) if pattern.startswith("%") else ("prefix", frag)


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
    tie_break: tuple[str, str] | None = None
    policy_order: bool = True

    def render(self) -> str:
        parts = [f"SELECT {self.select} FROM {self.table}"]
        if self.conds:
            parts.append("WHERE " + f" {self.conn or 'AND'} ".join(c.render(self.table) for c in self.conds))
        if self.order:
            primary_key = schema.TABLE_COLUMNS[self.table][0]
            tie = self.tie_break
            if tie is None and self.policy_order and self.order[0] != primary_key:
                tie = (primary_key, "ASC")
            parts.append(f"ORDER BY {self.order[0]} {self.order[1]}"
                         + (f", {tie[0]} {tie[1]}" if tie else ""))
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        return " ".join(parts)

    def cond_cols(self) -> set[str]:
        return {c.col for c in self.conds}

    def ops(self) -> list[str]:
        return [c.op for c in self.conds]


def _value_key(v):
    return (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))


def canonical_flat(q: Query) -> Query:
    """조건 순서를 정규형으로 맞춘다 (AND·OR: 스키마 컬럼 순서, IN 값: 값 순서). 조건이 1개 이하면 conn 은 None."""
    conds = tuple(sorted((Cond(c.col, c.op, tuple(sorted(c.val, key=_value_key))) if c.op == "IN" else c
                          for c in q.conds), key=lambda c: schema.col_index(q.table, c.col)))
    conn = q.conn if len(conds) >= 2 else None
    return replace(q, conds=conds, conn=conn)


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
    r"(?: ORDER BY (?P<ocol>\w+) (?P<odir>ASC|DESC)(?:, (?P<tcol>\w+) (?P<tdir>ASC|DESC))?)?"
    r"(?: LIMIT (?P<limit>\d+))?$")


def _parse_lit(s: str):
    if s.startswith("'"):
        return s[1:-1]
    return float(s) if "." in s else int(s)


def parse_flat(sql: str) -> Query:
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
    tie = (m["tcol"], m["tdir"]) if m["tcol"] else None
    return Query(m["table"], m["sel"], tuple(conds), conn, order, limit,
                 tie_break=tie, policy_order=False)


# ---------------------------------------------------------------------------
# 구조 규칙
# ---------------------------------------------------------------------------

def violations_flat(q: Query) -> list[str]:
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
        if k == "date":
            for value in ([c.val, c.val2] if c.op == "BETWEEN" else [c.val]):
                try:
                    valid = date.fromisoformat(value).isoformat() == value
                except (ValueError, TypeError):
                    valid = False
                if not valid:
                    v.append(f"Invalid ISO date: {value!r}")
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
    elif q.conn is not None:
        v.append("조건 1개 이하인데 연결어 있음")

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
        if ocol not in cols or (not schema.orderable(t, ocol) and ocol != cols[0]):
            v.append(f"정렬 불가 컬럼 {ocol}")
        if odir not in ("ASC", "DESC"):
            v.append(f"정렬 방향 {odir}")
        if ocol in eq_cols:
            v.append("= 조건으로 고정된 컬럼으로 정렬")
        if q.tie_break and (q.tie_break[0] != cols[0] or q.tie_break[1] not in ("ASC", "DESC")
                            or q.tie_break[0] == ocol):
            v.append("보조 정렬은 중복되지 않는 자기 기본키 ASC/DESC만 허용")
    elif q.tie_break:
        v.append("주 정렬 없는 보조 정렬")
    if q.limit is not None:
        if not q.order:
            v.append("LIMIT 은 ORDER BY 가 있을 때만")
        if not 1 <= q.limit <= MAX_LIMIT:
            v.append(f"LIMIT {q.limit} 범위 밖 (1~{MAX_LIMIT})")

    if not v and canonical(q).render() != q.render():
        v.append("정규형 순서가 아님")
    return v


# ---------------------------------------------------------------------------
# 직렬화 · 특징
# ---------------------------------------------------------------------------

def to_dict_flat(q: Query) -> dict:
    return {
        "sql": q.render(), "table": q.table, "select": q.select,
        "conds": [{"col": c.col, "op": c.op, "val": c.val, **({"val2": c.val2} if c.op == "BETWEEN" else {})}
                  for c in q.conds],
        "conn": q.conn,
        "order": {"col": q.order[0], "dir": q.order[1]} if q.order else None,
        "limit": q.limit,
        "tie_break": list(q.tie_break) if q.tie_break else None,
        "policy_order": q.policy_order,
        "skeleton": skeleton(q),
        **q.meta,
    }


def from_dict_flat(d: dict) -> Query:
    conds = tuple(Cond(c["col"], c["op"], tuple(c["val"]) if c["op"] == "IN" else c["val"], c.get("val2"))
                  for c in d["conds"])
    order = (d["order"]["col"], d["order"]["dir"]) if d["order"] else None
    tie = tuple(d["tie_break"]) if d.get("tie_break") else None
    policy_order = d.get("policy_order", False)
    if "policy_order" not in d and d.get("sql"):
        match = _SQL_RE.fullmatch(d["sql"])
        if match and match["tcol"]:
            tie = (match["tcol"], match["tdir"])
    return Query(d["table"], d["select"], conds, d["conn"], order, d["limit"],
                 tie_break=tie, policy_order=policy_order)


def limit_kind(limit: int | None) -> str:
    return "-" if limit is None else ("1" if limit == 1 else "n")


def skeleton_flat(q: Query) -> str:
    """리터럴을 뺀 구조 서명. 예: items|sel=ident|AND|category=,price<|ord=price:ASC|lim=n"""
    sk = schema.select_kind(q.table, q.select, q.cond_cols())
    conds = ",".join(f"{c.col}{c.op}" for c in q.conds) or "-"
    order = f"{q.order[0]}:{q.order[1]}" if q.order else "-"
    return f"{q.table}|sel={sk}|{q.conn or '-'}|{conds}|ord={order}|lim={limit_kind(q.limit)}"


def features_flat(q: Query) -> set[tuple[str, object]]:
    """구조 홀드아웃 조합을 판정할 원자 특징 집합."""
    f: set[tuple[str, object]] = {("table", q.table), ("n", len(q.conds)), ("conn", q.conn),
                                  ("order", q.order is not None), ("limit", q.limit is not None)}
    f |= {("op", c.op) for c in q.conds}
    return f


AGG_PATTERN = r"(?:COUNT\(\*\)|(?:SUM|AVG|MIN|MAX)\(\w+\))"
GROUP_COLUMNS = {
    "customers": ("city", "membership"),
    "items": ("category",),
    "orders": ("customer_id", "item_id", "status", "order_date"),
}
NUMERIC_COLUMNS = {t: tuple(c for c in cols if schema.kind(t, c) == "num")
                   for t, cols in schema.TABLE_COLUMNS.items()}


@dataclass(frozen=True)
class AdvancedQuery:
    table: str
    aggregate: str | None = None
    conds: tuple = ()
    group_by: str | None = None
    having: tuple[str, float] | None = None
    order: tuple[str, str] | None = None
    limit: int | None = None
    # (outer column, comparison, inner aggregate), always the same table.
    subquery: tuple[str, str, str] | None = None
    inner_conds: tuple = ()
    projection: str = "*"
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def select(self):
        if self.aggregate:
            return f"{self.group_by}, {self.aggregate}" if self.group_by else self.aggregate
        return self.projection

    @property
    def conn(self):
        return "AND" if len(self.conds) + bool(self.subquery) >= 2 else None

    def cond_cols(self):
        return {c.col for c in self.conds} | ({self.subquery[0]} if self.subquery else set())

    def ops(self):
        return [c.op for c in self.conds] + ([self.subquery[1]] if self.subquery else [])

    def render(self):
        parts = [f"SELECT {self.select} FROM {self.table}"]
        conditions = [c.render(self.table) for c in self.conds]
        if self.subquery:
            col, op, aggregate = self.subquery
            inner = f"SELECT {aggregate} FROM {self.table}"
            if self.inner_conds:
                inner += " WHERE " + " AND ".join(c.render(self.table) for c in self.inner_conds)
            conditions.append(f"{col} {op} ({inner})")
        if conditions:
            parts.append("WHERE " + " AND ".join(conditions))
        if self.group_by:
            parts.append(f"GROUP BY {self.group_by}")
        if self.having:
            op, value = self.having
            parts.append(f"HAVING {self.aggregate} {op} {float(value):.6g}")
        if self.order:
            parts.append(f"ORDER BY {self.order[0]} {self.order[1]}, {self.group_by} ASC")
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        return " ".join(parts)


def _canonical_aggregate(query):
    return replace(query,
                   conds=canonical_flat(Query(query.table, "*", query.conds)).conds,
                   inner_conds=canonical_flat(Query(query.table, "*", query.inner_conds)).conds)


_GROUP_RE = re.compile(
    rf"^SELECT (?:(?P<key>\w+), )?(?P<agg>{AGG_PATTERN}) FROM (?P<table>\w+)"
    rf"(?: WHERE (?P<where>.+?))?(?: GROUP BY (?P<group>\w+))?"
    rf"(?: HAVING (?P<hagg>{AGG_PATTERN}) (?P<hop><=|>=|!=|=|<|>) (?P<hval>-?\d+(?:\.\d+)?))?"
    rf"(?: ORDER BY (?P<order>{AGG_PATTERN}) (?P<dir>ASC|DESC), (?P<tie>\w+) ASC)?"
    r"(?: LIMIT (?P<limit>\d+))?$")
_SUB_RE = re.compile(
    rf"^SELECT (?P<select>\*|\w+) FROM (?P<table>\w+) WHERE (?P<prefix>.*?)"
    rf"(?P<col>\w+) (?P<op><=|>=|=|<|>) \(SELECT (?P<agg>{AGG_PATTERN}) FROM (?P<inner>\w+)"
    r"(?: WHERE (?P<where>.+))?\)$")


def _conditions(table, text):
    if not text:
        return ()
    query = parse_flat(f"SELECT * FROM {table} WHERE {text}")
    if query.conn == "OR":
        raise ValueError("Advanced conditions only support AND")
    return query.conds


def _parse_aggregate(sql):
    if "(SELECT " in sql:
        match = _SUB_RE.fullmatch(sql)
        if not match or match["inner"] != match["table"]:
            raise ValueError("Unsupported scalar subquery")
        prefix = match["prefix"]
        if prefix and not prefix.endswith(" AND "):
            raise ValueError("Invalid outer conditions")
        query = AdvancedQuery(match["table"], conds=_conditions(match["table"], prefix[:-5]),
                              subquery=(match["col"], match["op"], match["agg"]),
                              inner_conds=_conditions(match["table"], match["where"]),
                              projection=match["select"])
    else:
        match = _GROUP_RE.fullmatch(sql)
        if not match or match["key"] != match["group"]:
            raise ValueError("Unsupported aggregate SQL")
        if match["hagg"] and match["hagg"] != match["agg"]:
            raise ValueError("HAVING must use the selected aggregate")
        if match["order"] and (match["order"] != match["agg"] or match["tie"] != match["group"]):
            raise ValueError("Invalid aggregate ordering")
        query = AdvancedQuery(match["table"], match["agg"], _conditions(match["table"], match["where"]),
                              match["group"], (match["hop"], float(match["hval"])) if match["hop"] else None,
                              (match["order"], match["dir"]) if match["order"] else None,
                              int(match["limit"]) if match["limit"] else None)
    problems = _violations_aggregate(query)
    if problems:
        raise ValueError("; ".join(problems))
    return query


def _violations_aggregate(query):
    errors = []
    if query.table not in schema.TABLE_COLUMNS:
        return ["Unknown table"]
    if bool(query.aggregate) == bool(query.subquery):
        errors.append("Exactly one aggregate or subquery is required")
    for conditions in (query.conds, query.inner_conds):
        if len(conditions) > 2:
            errors.append("At most two row conditions")
        if conditions:
            # Reuse literal/operator validation without projection-specific constraints.
            flat = Query(query.table, "*", conditions, "AND" if len(conditions) > 1 else None)
            # Question labels may retain IN mention order; identity is canonical.
            errors.extend(violations_flat(canonical_flat(flat)))
    if query.aggregate:
        match = re.fullmatch(r"(COUNT|SUM|AVG|MIN|MAX)\((\*|\w+)\)", query.aggregate)
        if not match:
            return errors + ["Invalid aggregate"]
        function, col = match.groups()
        if function == "COUNT":
            if col != "*":
                errors.append("COUNT uses *")
        elif col not in NUMERIC_COLUMNS[query.table] and not (
                function in ("MIN", "MAX") and col == "order_date" and query.table == "orders"):
            errors.append("Invalid aggregate column")
        if query.group_by and query.group_by not in GROUP_COLUMNS[query.table]:
            errors.append("Invalid group key")
        if query.having and (not query.group_by or col == "order_date"):
            errors.append("HAVING requires a numeric aggregate and GROUP BY")
        if query.order and (not query.group_by or query.order[0] != query.aggregate
                            or query.order[1] not in ("ASC", "DESC")):
            errors.append("Invalid aggregate order")
        if query.limit is not None and (not query.order or not 1 <= query.limit <= 10):
            errors.append("LIMIT requires group ordering, range 1..10")
    if query.having and query.having[0] not in ("=", "!=", "<", ">", "<=", ">="):
        errors.append("Invalid HAVING operator")
    if query.subquery:
        col, op, aggregate = query.subquery
        if col not in NUMERIC_COLUMNS[query.table] or op not in ("=", "<", ">", "<=", ">="):
            errors.append("Invalid scalar comparison")
        if aggregate not in [f"{fn}({col})" for fn in ("AVG", "MIN", "MAX")]:
            errors.append("Subquery uses AVG/MIN/MAX of the same numeric column")
        if query.projection != "*" and query.projection not in schema.TABLE_COLUMNS[query.table]:
            errors.append("Invalid projection")
        if query.group_by or query.having or query.order or query.limit is not None:
            errors.append("Subquery output grouping/ordering is outside initial scope")
    return errors


def _to_dict_aggregate(query):
    def conditions(rows):
        return [{"col": c.col, "op": c.op, "val": c.val,
                 **({"val2": c.val2} if c.op == "BETWEEN" else {})} for c in rows]
    return {"sql": query.render(), "query_kind": "subquery" if query.subquery else "group" if query.group_by else "aggregate",
            "table": query.table, "select": query.select, "aggregate": query.aggregate,
            "conds": conditions(query.conds), "conn": query.conn, "group_by": query.group_by,
            "having": list(query.having) if query.having else None,
            "order": {"col": query.order[0], "dir": query.order[1]} if query.order else None,
            "limit": query.limit, "subquery": list(query.subquery) if query.subquery else None,
            "inner_conds": conditions(query.inner_conds), "projection": query.projection, **query.meta}


def _from_dict_aggregate(record):
    def conditions(rows):
        return tuple(Cond(c["col"], c["op"], tuple(c["val"]) if c["op"] == "IN" else c["val"], c.get("val2")) for c in rows)
    order = record.get("order")
    return AdvancedQuery(record["table"], record.get("aggregate"), conditions(record["conds"]),
                         record.get("group_by"), tuple(record["having"]) if record.get("having") else None,
                         (order["col"], order["dir"]) if order else None, record.get("limit"),
                         tuple(record["subquery"]) if record.get("subquery") else None,
                         conditions(record.get("inner_conds", [])), record.get("projection", "*"))


def parse(sql):
    if re.match(r"SELECT (?:\w+, )?(?:COUNT|SUM|AVG|MIN|MAX)\(", sql) or "(SELECT " in sql:
        return _parse_aggregate(sql)
    return parse_flat(sql)


def canonical(query):
    return _canonical_aggregate(query) if isinstance(query, AdvancedQuery) else canonical_flat(query)


def violations(query):
    return _violations_aggregate(query) if isinstance(query, AdvancedQuery) else violations_flat(query)


def to_dict(query):
    return _to_dict_aggregate(query) if isinstance(query, AdvancedQuery) else to_dict_flat(query)


def from_dict(record):
    return _from_dict_aggregate(record) if record.get("query_kind") else from_dict_flat(record)


def skeleton(query):
    if isinstance(query, AdvancedQuery):
        return f"{query.table}|{query.aggregate}|group={query.group_by}|having={bool(query.having)}|sub={query.subquery}"
    return skeleton_flat(query)


def features(query):
    result = features_flat(query)
    if isinstance(query, AdvancedQuery):
        result |= {("group", bool(query.group_by)), ("having", bool(query.having)),
                   ("subquery", bool(query.subquery)), ("aggregate", query.aggregate)}
    return result
