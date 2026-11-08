# 역할: 3단계 질문과 SQL의 라벨 규칙 위반·모호성을 검사한다.
# 사용처: 3단계 학습 데이터 정제와 평가의 규칙 일치 여부 판정에 사용한다.

"""3단계 라벨 규칙 판정: 일반 조회는 2단계 판정(L1~L4·L14)에 L19 수량 합계를 더하고, 집계·그룹·서브쿼리(고급)는 L19~L24로 판정한다.

2단계 판정처럼 질문의 단서로 정답이 규칙과 **어긋나는 것이 분명한 경우**만 잡는다. 단서가 없거나 여러 해석의 단서가 함께 있으면
미검사 항목이 있으면 review로 남긴다. drop_reason은 확정 위반만 노출한다. 판정은 기본 해석 표(docs/text-to-sql-label-rules.md L19~L25) 기준이다.
  L19 집계 함수   — 정답 집계 함수의 단서가 질문에 없다(그룹 집계는 함수를 생략한 기본 표현도 인정, 라벨 규칙 3-14).
                    집계 표현("average price")이 다른 함수와만 이어진다. "how many units"는 COUNT 단서가 아니다.
                    orders COUNT에 고객·상품 수 단서만 있거나, customers·items COUNT·SUM(quantity)에 주문 건수 단서가 있다.
                    일반 조회의 stock·quantity가 여러 행인데 how many units / how much stock 이다(합계 SUM)
  L20 집계 컬럼   — 집계·서브쿼리 비교 컬럼의 단서가 질문에 없거나, 집계 표현이 다른 컬럼과만 이어진다
  L21 그룹        — GROUP BY 인데 그룹 키 단서가 없다. 그룹이 없는데 "per category", "grouped by" 처럼 그룹을 요구한다
  L22 조건 범위   — 질문의 조건 값(범주 값·숫자·날짜)이 SQL 에 없다(필터 누락)
  L23 기준 집단   — 서브쿼리 안쪽 조건의 값(ID·이름·날짜·숫자)이 질문에 나오지 않는다(바깥 조건과 같은 값은 두 번). 안쪽 조건이
                    없는데 "their average" 처럼 앞의 집합을 가리키거나, SQL 에 없는 조건 값으로 기준 집단을 말한다.
                    출력 컬럼은 L2·L3·L14 판정을 쓴다
  L24 정렬·LIMIT  — 그룹 정렬을 요구하지 않았거나 방향 단서가 반대다. 정렬이 없는데 정렬을 요구한다. LIMIT n 의 개수가 질문에
                    없다. LIMIT 이 없는데 top 3 / which category has the highest 처럼 개수를 정한다

비교는 값·컬럼에 붙은 방향을 읽고, 행 조건과 집계 조건 및 서브쿼리 기준 집단을 구분한다.
범주 값의 단순 횟수 세기는 하지 않으며, 명시된 기준 구간에 값을 연결한다. 해석하지 못한 범위·방향은 unchecked에 기록한다.
단서 목록은 라벨 규칙 문서와 학습 문장(조합기 틀·유사어 사전, LLM 학습 쌍)에서 정한다. 평가 오답을 보고 고치지 않는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import partial, lru_cache
from typing import Literal

from src.data import db_gen
from src.data.stage2 import clean_ambiguous as base
from src.data.stage3 import schema, sql_ast

REASONS = {**base.REASONS, "L19": "L19 집계 함수", "L20": "L20 집계 컬럼", "L21": "L21 그룹", "L22": "L22 조건 범위",
           "L23": "L23 서브쿼리 기준 집단", "L24": "L24 정렬·LIMIT"}
_strip_conditions = partial(base._strip_conditions, schema_module=schema)
select_label = base.select_label

# ---------------------------------------------------------------------------
# 단서
# ---------------------------------------------------------------------------

_COUNTED = r"(?:customers|clients|orders|items|products|purchases)"
# L19: 집계 함수 단서. "how many units"는 행 개수가 아니라 수량 합계다
FUNCTION_CUE = {
    "COUNT": re.compile(r"\bhow many\b(?!\s+(?:total\s+)?(?:units?|pieces)\b)|\bcount(?:s|ed|ing)?\b|\bnumber of\b(?!\s+units?\b)"
                        r"|\btall(?:y|ies)\b|\btotal (?:number|count)\b|\btotal (?:orders|items|products|customers|clients"
                        r"|records|rows)\b|\b(?:customer|client|order|item|product)s? totals?\b"
                        rf"|\b\d+\s+(?:\w+\s+)?{_COUNTED}\b"),
    "SUM": re.compile(r"\b(?:total(?:s|ed|ing)?|sum(?:s|med)?|combined|altogether|in all|add(?:ed)? up|overall)\b"
                      r"|\bhow many (?:total )?(?:units?|pieces)\b|\bhow much stock\b|\bunits ordered\b"),
    "AVG": re.compile(r"\b(?:average[sd]?|averaging|avg|mean|means|typical(?:ly)?)\b"),
    "MIN": re.compile(r"\b(?:min|minimum|minimums|minima|lowest|smallest|least|fewest|cheapest|earliest|oldest|first"
                      r"|bottom|lower bound)\b|\bthan (?:every|all|any)\b"),
    "MAX": re.compile(r"\b(?:max|maximum|maximums|maxima|highest|largest|biggest|greatest|most|top|priciest|latest"
                      r"|newest|recent|last|peak|upper bound|lead(?:s|ing)?)\b|\bthan (?:every|all|any)\b"),
}
# 집계 표현(함수 단어와 컬럼 단어가 3단어 안에 이어짐)
_FN_WORDS = {
    "SUM": r"total(?:s|ed)?|sum(?:s|med)?|combined",
    "AVG": r"average[sd]?|averaging|avg|mean",
    "MIN": r"min|minimum|minima|lowest|smallest|least",
    "MAX": r"max|maximum|maxima|highest|largest|biggest|greatest",
}
_COL_WORDS = {
    "price": r"prices?|priced|pricing|costs?",
    "stock": r"stocks?|inventory|on hand",
    "quantity": r"quantit(?:y|ies)|qty|units?|order sizes?",
    "order_date": r"(?:order )?dates?",
}
_SPECIAL_EXPRESSION = {
    ("MIN", "price"): re.compile(r"\bcheapest\b"), ("MAX", "price"): re.compile(r"\b(?:priciest|most expensive)\b"),
    ("MIN", "order_date"): re.compile(r"\b(?:earliest|oldest|first order)\b"),
    ("MAX", "order_date"): re.compile(r"\b(?:latest|newest|most recent|last order)\b"),
}
EXPRESSION = {
    (fn, col): re.compile(rf"\b(?:{fw})\b(?: of)?(?: the)?(?: [\w']+)?\s+(?:{cw})\b|\b(?:{cw})\s+(?:{fw})\b")
    for fn, fw in _FN_WORDS.items() for col, cw in _COL_WORDS.items()
}
# L20: 집계 컬럼 단서
COLUMN_CUE = {
    "price": re.compile(r"\$|\b(?:price[sd]?|pricing|pricey|priciest|cost(?:s|ing|lier|liest)?|expensive|cheap(?:er|est)?"
                        r"|dollars?)\b"),
    "stock": re.compile(r"\b(?:stock(?:ed|s)?|inventory|on hand|left|available|availability|supply|units?)\b"),
    "quantity": re.compile(r"\b(?:quantit(?:y|ies)|qty|units?|pieces|size|sizes|amount|volume)\b"),
    "order_date": re.compile(r"\b(?:dates?|dated|days?|earliest|latest|oldest|newest|recent|first|last|when)\b"),
}
# 집계 함수를 생략한 그룹 표현의 기본값(라벨 규칙 3-14): 고객·상품·주문 → COUNT(*), price → AVG(price),
# stock → SUM(stock), units ordered → SUM(quantity)
IMPLICIT_GROUP = {
    ("customers", "COUNT(*)"): re.compile(r"\b(?:customers|clients|members|patrons|shoppers|people)\b"),
    ("items", "COUNT(*)"): re.compile(r"\b(?:items|products|things|articles)\b"),
    ("orders", "COUNT(*)"): re.compile(r"\b(?:orders|purchases|transactions)\b"),
    ("items", "AVG(price)"): COLUMN_CUE["price"],
    ("items", "SUM(stock)"): COLUMN_CUE["stock"],
    ("orders", "SUM(quantity)"): re.compile(r"\bunits ordered\b|\bquantit(?:y|ies)\b"),
}
ORDER_NOUN = re.compile(r"\b(?:orders?|purchases?|transactions?|placed|bought|ordered)\b")
PEOPLE_OR_ITEMS_COUNT = re.compile(r"\b(?:how many|number of)\s+(?:\w+\s+){0,2}?(?:customers|clients|items|products)\b")
ORDERS_COUNT = re.compile(r"\b(?:how many|number of|count of)\s+(?:\w+\s+){0,2}?orders\b")
# 비교·개수 표현의 least / most / first / max 는 집계 함수 단서가 아니다 ("at least 5", "max $60.00", "the first 3 groups")
COMPARISON_WORDS = re.compile(r"\bat (?:least|most)\b|\bno (?:less|more|fewer) than\b|\b(?:less|more|fewer) than\b(?! (?:every|all|any)\b)"
                              r"|\b(?:only )?the first \d+ \w+|\b(?:first|top) \d+\b"
                              r"|\b(?:max|min)\.?\s*\$?\d|\$?\d+(?:\.\d+)?(?:\s+\w+)?\s+(?:max|min|maximum|minimum)\b")
# 일반 조회 stock·quantity 의 수량 합계 단서. "each item", "per order"는 행마다 값을 묻는 말이라 합계가 아니다
UNITS_TOTAL = re.compile(r"\bhow many (?:units|pieces)\b|\bhow much stock\b")
PER_ROW = re.compile(r"\b(?:each|every|per|individual(?:ly)?|respective)\b")

# L21: 그룹 키 단서와 그룹 요구 표현
GROUP_KEY_CUE = {
    "city": r"cit(?:y|ies)|locations?|towns?",
    "membership": r"memberships?|tiers?|levels?",
    "category": r"categor(?:y|ies)|kinds?|types?|departments?",
    "customer_id": r"customers?|clients?|buyers?|shoppers?|purchasers?",
    "item_id": r"items?|products?|articles?",
    "status": r"status(?:es)?|stages?|states?",
    "order_date": r"dates?|days?|daily",
}
GROUP_KEY_RE = {k: re.compile(rf"\b(?:{v})\b") for k, v in GROUP_KEY_CUE.items()}
# 그룹 요청: "per / for each / by" 뒤의 말이 **그 테이블의 허용 그룹 키**일 때만 그룹 요청이다. 그래서 items 의
# "price per item"(단가), orders 의 "units per order"(주문당 수량)는 그룹 요청이 아니고, orders 의 "units ordered per item"은
# 그룹 요청이다. "by customer 1452"처럼 뒤에 값이 오면 조건이다
_GROUP_KEY_LABELS = {"customer_id": r"customer ids?", "item_id": r"item ids?", "status": r"order status(?:es)?",
                     "order_date": r"order dates?", "membership": r"membership (?:tiers?|levels?)"}
_GROUP_WORDS = r"\bgroup(?:s|ed|ing)?\b|\bbreakdown\b|\bbreak(?:s|ing)? (?:\w+ ){0,4}?down\b|\bbroken (?:down|out)\b|\bpartition"


def _group_request(table: str) -> re.Pattern:
    keys = "|".join(alt for k in sql_ast.GROUP_COLUMNS[table]
                    for alt in filter(None, (_GROUP_KEY_LABELS.get(k), GROUP_KEY_CUE[k])))
    return re.compile(rf"\b(?:per|for each|for every|by|grouped by|broken (?:down|out) by)\s+(?:the\s+|each\s+)?"
                      rf"(?:{keys})\b(?!\s*(?:ids?|numbers?|no\.?)?\s*#?\s*\d)"
                      rf"|{_GROUP_WORDS}"
                      rf"|\beach (?:{keys})\b(?!\s*(?:ids?|numbers?|no\.?)?\s*#?\s*\d)"
                      rf"|\b(?:{keys}) (?:and|with|along with) (?:their|its|each|the)\b")


GROUP_REQUEST = {table: _group_request(table) for table in sql_ast.GROUP_COLUMNS}

# L23: 앞에서 지정한 집합을 기준 집단으로 다시 가리키는 말
_AGG_WORD = r"(?:average|mean|avg|minimum|min|lowest|smallest|maximum|max|highest|largest)"
BACK_REFERENCE = re.compile(
    rf"\b(?:their|those|these|that group's|the group's|this group's|that set's)\s+(?:own\s+)?{_AGG_WORD}\b"
    rf"|\b{_AGG_WORD}\b[^.;?]{{0,30}}?\b(?:of|among|across|for|within|from)\s+"
    r"(?:them|those|these|that group|that set|the same (?:group|set))\b")
# L20 기본값: 숫자 없는 low stock 은 stock <= 10
LOW_STOCK = re.compile(r"\blow (?:on )?stock\b|\brunning low\b|\bstock is low\b")
CATEGORICAL_VALUES = {"customers": db_gen.CITIES + db_gen.MEMBERSHIPS, "items": db_gen.CATEGORIES,
                      "orders": db_gen.STATUSES}

# L24: 정렬 방향·정렬 요구·개수
_HIGH = (r"highest|higher|largest|larger|biggest|bigger|most|more|greatest|greater|top|busiest|busier|latest|later"
         r"|newest|recent|priciest|pricier")
_LOW = r"lowest|lower|smallest|smaller|fewest|fewer|least|less|bottom|earliest|earlier|oldest|older|cheapest|cheaper"
DESC_CUE = re.compile(rf"\b(?:desc|descending|decreasing|{_HIGH}|max|maximum|high to low|best|lead(?:s|ing)?)\b")
ASC_CUE = re.compile(rf"\b(?:asc|ascending|increasing|{_LOW}|min|minimum|low to high|worst|sort|sorted|sorting"
                     r"|order(?:ed)? by|rank(?:ed|ing)?|in order)\b")
DESC_ONLY = re.compile(r"\b(?:descending|decreasing|high(?:est)? to low(?:est)?|(?:latest|newest) to (?:earliest|oldest)"
                       rf"|(?:{_HIGH})(?: \w+){{0,3}} first|starting (?:with|from) the (?:{_HIGH})"
                       rf"|from (?:the )?(?:{_HIGH})(?: \w+){{0,2}} (?:to|down(?:ward)?)\b"
                       rf"|(?:{_HIGH})(?: \w+){{0,2}} to (?:the )?(?:{_LOW}))\b")
ASC_ONLY = re.compile(r"\b(?:ascending|increasing|low(?:est)? to high(?:est)?|(?:earliest|oldest) to (?:latest|newest)"
                      rf"|(?:{_LOW})(?: \w+){{0,3}} first|starting (?:with|from) the (?:{_LOW})"
                      rf"|from (?:the )?(?:{_LOW})(?: \w+){{0,2}} (?:to|up(?:ward)?)\b"
                      rf"|(?:{_LOW})(?: \w+){{0,2}} to (?:the )?(?:{_HIGH}))\b")
# 동점 처리 설명("status ascending for ties", "break ties by …")은 정렬 방향 단서가 아니다
TIE_CLAUSE = re.compile(r"\b(?:and |with |using )?(?:\w+ ){0,3}?(?:for|on|when|if|in case of|to break) (?:ties|tied|a tie)\b"
                        r"|\bbreak(?:ing)? ties (?:by|with|on|using) (?:\w+ ){0,2}(?:ascending|descending|asc|desc)?"
                        r"|\bties? (?:broken|resolved) by (?:\w+ ){0,2}(?:ascending|descending|asc|desc)?")
SORT_REQUEST = re.compile(r"\b(?:sort|sorted|sorting|rank|ranked|ranking|arranged?|order by)\b"
                          r"|\bin (?:ascending|descending|increasing|decreasing) order\b|\bin order of\b")
NUMBER_WORDS = {1: "one|single", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                9: "nine", 10: "ten"}
_KEY_NOUN = r"(?:category|city|tier|membership|status|customer|item|product|date|day|customer id|item id)"
SINGLE_CUE = re.compile(r"\b(?:1|one|single|top|first|only|just|lead(?:s|ing)?"
                        r"|the (?:\w+est|most|least|fewest|max(?:imum)?|min(?:imum)?))\b"
                        rf"|\b(?:which|the) (?:\w+ ){{0,2}}?{_KEY_NOUN}\b(?! ids?\b)")
_COUNT_WORD = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"
LIMIT_REQUEST = re.compile(rf"\btop {_COUNT_WORD}\b|\b(?:only|just) (?:the )?(?:first|top) {_COUNT_WORD}\b"
                           rf"|\b(?:first|bottom) {_COUNT_WORD} (?:groups|categories|cities|tiers|statuses|customers|items|dates)\b"
                           rf"|\bwhich (?:\w+ ){{0,2}}?{_KEY_NOUN} (?:has|had|leads|shows|sees|saw|is)\b"
                           rf"|\b(?:single|only one|just one)\b (?:\w+ ){{0,2}}?(?:{_KEY_NOUN}|group)\b"
                           rf"|\bthe (?:top|leading|busiest) {_KEY_NOUN}\b(?!s)")


# ---------------------------------------------------------------------------
# 값
# ---------------------------------------------------------------------------

def _value_count(question: str, table: str, col: str, value) -> int:
    """질문에 값이 나온 횟수. price 는 50 / 50.00 / $50, 문자열 LIKE 는 % 를 뗀 조각."""
    if isinstance(value, str):
        text = value.strip("%")
        return len(re.findall(rf"(?<![a-z0-9]){re.escape(text)}(?![a-z0-9])", question)) if text else 0
    forms = {str(value), schema.fmt_literal(table, col, value)}
    if isinstance(value, float) and value.is_integer():
        forms.add(str(int(value)))
    pattern = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return len(re.findall(rf"(?<![\w.]){pattern}(?![\w]|\.\d)", question))


def _cond_values(cond):
    return cond.val if cond.op == "IN" else (cond.val, cond.val2) if cond.op == "BETWEEN" else (cond.val,)


def _extra_value(q: str, g: sql_ast.AdvancedQuery) -> bool:
    """질문에 SQL 에 없는 범주 값·숫자·날짜가 있다(조건 누락)."""
    values = [v for c in (*g.conds, *g.inner_conds) for v in _cond_values(c)]
    values += [g.having[1]] if g.having else []
    values += [g.limit] if g.limit is not None else []
    strings = {v.strip("%") for v in values if isinstance(v, str)}
    numbers = {float(v) for v in values if isinstance(v, (int, float))}
    if any(v not in strings and re.search(rf"(?<![a-z]){re.escape(v)}(?![a-z])", q)
           for v in CATEGORICAL_VALUES[g.table]):
        return True
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    if any(d not in strings for d in dates):
        return True
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", q)
    return any(float(n) not in numbers for n in re.findall(r"(?<![\w.#-])(\d+(?:\.\d+)?)(?![\w.]*\d)", text))


def _aggregate_parts(expression: str) -> tuple[str, str]:
    function, col = expression[:-1].split("(")
    return function, col


def _expression(q: str, function: str, col: str) -> bool:
    special = _SPECIAL_EXPRESSION.get((function, col))
    return bool((special and special.search(q))
                or (function in _FN_WORDS and col in _COL_WORDS and EXPRESSION[(function, col)].search(q)))


def _strip_sort(q: str) -> str:
    q = TIE_CLAUSE.sub(" ", q)
    return COMPARISON_WORDS.sub(" ", ASC_ONLY.sub(" ", DESC_ONLY.sub(" ", q)))


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------

def _function_reason(q: str, function: str, col: str, table: str, grouped: bool) -> str | None:
    qf = _strip_sort(q)
    implicit = IMPLICIT_GROUP.get((table, f"{function}({col})")) if grouped else None
    others = [f for f in FUNCTION_CUE if f != function and f != "COUNT"]
    implicit_ok = implicit and implicit.search(qf) and not any(FUNCTION_CUE[f].search(qf) for f in others)
    if not FUNCTION_CUE[function].search(q) and not implicit_ok:
        return "L19"
    if col != "*" and not _expression(qf, function, col):
        if any(_expression(qf, f, col) for f in _FN_WORDS if f != function):
            return "L19"
        columns = [c for c in (*sql_ast.NUMERIC_COLUMNS[table], "order_date") if c != col and c in COLUMN_CUE]
        if any(_expression(qf, function, c) for c in columns
               if c != "order_date" or (table == "orders" and function in ("MIN", "MAX"))):
            return "L20"
    if function == "COUNT":
        if table == "orders" and PEOPLE_OR_ITEMS_COUNT.search(q) and not ORDER_NOUN.search(q):
            return "L19"
        if table != "orders" and ORDERS_COUNT.search(q):
            return "L19"
    elif function == "SUM" and ORDERS_COUNT.search(q) and not COLUMN_CUE["quantity"].search(q):
        return "L19"
    if col != "*" and not COLUMN_CUE[col].search(q):
        return "L20"
    return None


def _subquery_reason(q: str, g: sql_ast.AdvancedQuery, pair: dict) -> str | None:
    raw = pair["question"]
    col, _, inner = g.subquery
    function, _ = _aggregate_parts(inner)
    reason = _function_reason(q, function, col, g.table, False)
    if reason:
        return reason
    back = BACK_REFERENCE.search(q)
    if not g.inner_conds and g.conds and back:
        return "L23"
    if not g.inner_conds and _extra_value(q, g):
        return "L23"
    if g.inner_conds and not back:
        need: dict[tuple[str, object], int] = {}
        for c in (*g.conds, *g.inner_conds):
            for v in _cond_values(c):
                need[(c.col, v)] = need.get((c.col, v), 0) + 1
        # 범주 값은 clothes / non-clothing 처럼 다른 표기로도 써서 세지 않는다. 0 은 sold out 처럼 말로 쓴다
        for c in g.inner_conds:
            if schema.kind(g.table, c.col) == "cat":
                continue
            if c.col == "stock" and c.op == "<=" and c.val == 10 and LOW_STOCK.search(q):
                continue
            for v in _cond_values(c):
                if v != 0 and _value_count(raw, g.table, c.col, v) < need[(c.col, v)]:
                    return "L23"
    # 출력 컬럼(L2·L3·L14)은 바깥 조건만 둔 일반 조회로 판정한다. SELECT=WHERE 존재 확인(L1)은 서브쿼리에 적용하지 않는다(L23)
    flat = sql_ast.Query(g.table, g.projection, g.conds, "AND" if len(g.conds) > 1 else None).render()
    reason = base.drop_reason(dict(pair, sql=flat), stage=3)
    return None if reason == "L1" else reason


def _group_reason(q: str, g: sql_ast.AdvancedQuery) -> str | None:
    if g.group_by is None:
        return "L21" if GROUP_REQUEST[g.table].search(q) else None
    if not GROUP_KEY_RE[g.group_by].search(q):
        return "L21"
    q = TIE_CLAUSE.sub(" ", q)
    sort_asked = SORT_REQUEST.search(q) or DESC_ONLY.search(q) or ASC_ONLY.search(q)
    if g.order:
        if g.limit is None and not sort_asked:
            return "L24"
        if g.order[1] == "DESC" and (not DESC_CUE.search(q) or (ASC_ONLY.search(q) and not DESC_ONLY.search(q))):
            return "L24"
        if g.order[1] == "ASC" and (not ASC_CUE.search(q) or (DESC_ONLY.search(q) and not ASC_ONLY.search(q))):
            return "L24"
    elif sort_asked:
        return "L24"
    if g.limit is not None:
        if not re.search(rf"\b(?:{g.limit}|{NUMBER_WORDS.get(g.limit, g.limit)})\b", q) and not (
                g.limit == 1 and SINGLE_CUE.search(q)):
            return "L24"
    elif LIMIT_REQUEST.search(ASC_ONLY.sub(" ", DESC_ONLY.sub(" ", q))):
        return "L24"
    return None


def _mask_names(q: str, g: sql_ast.AdvancedQuery) -> str:
    """이름·LIKE 조각("steel partition", "max…")이 단서로 읽히지 않도록 가린다. 범주 값은 남긴다."""
    for c in (*g.conds, *g.inner_conds):
        if schema.kind(g.table, c.col) in ("name", "like") or c.op == "LIKE" or c.col in ("name", "item_name"):
            for v in _cond_values(c):
                text = str(v).strip("%")
                if text:
                    def mask(match):
                        # LIKE 'no%'의 no가 다른 조건의 no cheaper than까지 지우면 비교 방향이 바뀐다.
                        before = q[max(0, match.start()-55):match.start()]
                        name_context = re.search(r"\b(?:names?|named|called|like|contains?|containing|starts? with|ends? with|beginning with|ending with)\s*['\"]?$", before)
                        if c.op == "LIKE" and not name_context and re.match(rf"(?:{_CMP_ALT})\b", q[match.start():]):
                            return match.group()
                        return " value "
                    q = re.sub(rf"(?<![a-z0-9]){re.escape(text)}(?![a-z0-9])", mask, q)
    return q


def advanced_reason(pair: dict, g: sql_ast.AdvancedQuery) -> str | None:
    raw = pair["question"].lower().strip().replace("_", " ")
    q = _mask_names(raw, g)
    if g.subquery:
        return _subquery_reason(q, g, dict(pair, question=raw))
    function, col = _aggregate_parts(g.aggregate)
    reason = _function_reason(q, function, col, g.table, g.group_by is not None) or _group_reason(q, g)
    if reason:
        return reason
    return "L22" if _extra_value(q, g) else None


def _unique_row(g: sql_ast.Query) -> bool:
    keys = {schema.TABLE_COLUMNS[g.table][0]} | {"items": {"item_name"}, "customers": {"name"}}.get(g.table, set())
    return g.limit == 1 or any(c.col in keys and c.op == "=" for c in g.conds)


def _legacy_reason(pair: dict, g) -> str | None:
    """정답이 라벨 규칙과 어긋나면 사유 키(REASONS), 아니면 None. 3단계 문법 밖 SQL 은 판정하지 않는다."""
    if isinstance(g, sql_ast.AdvancedQuery):
        return advanced_reason(pair, g)
    reason = base.drop_reason(pair, stage=3)
    if reason:
        return reason
    q = pair["question"].lower()
    if g.select in ("stock", "quantity") and not _unique_row(g) and UNITS_TOTAL.search(q) and not PER_ROW.search(q):
        return "L19"
    return None


@dataclass
class Judgment:
    """위반·검토 필요·통과를 구분한다. 미검사 항목은 위반 사유로 쓰지 않는다."""
    status: Literal["violation", "review", "pass"]
    reason: str | None = None
    evidence: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)


@lru_cache(maxsize=16384)
def _parse_query(sql: str):
    """같은 SQL의 여러 질문·판정에서 불변 AST를 재사용한다."""
    return sql_ast.parse(sql)


# 비교 표현은 값 바로 앞·뒤에 붙은 것만 읽는다. 다른 조건이나 정렬의 단어를 빌려오지 않는다.
_CMP_PHRASES = {
    ">=": ("greater than or equal to", "higher than or equal to", "at or above", "meets or exceeds",
           "at least", "no less than", "no fewer than", "no cheaper than", "no lower than", "no smaller than", "not under", "not below", "meet or exceed", "meeting or exceeding", "reaches or exceeds",
           "on or after", "at or after", "at least equal to", "at least at", "equal or exceed", "match or exceed", "reach or exceed", "no earlier than", "not before", "starting at", "starting on", "since", "min", "minimum"),
    "<=": ("less than or equal to", "lower than or equal to", "at or below", "at or before", "at most equal to", "don't exceed", "without exceeding", "without going above", "price cap of", "doesn't exceed", "does not exceed", "do not exceed",
           "not greater than", "not higher than", "capped at",
           "no greater than", "no higher than", "at most", "no more than", "no later than", "not over",
           "not above", "not after", "on or before", "up to and including", "up to", "until", "through", "max", "maximum"),
    ">": ("more expensive than", "more than", "greater than", "higher than", "larger than", "pricier than",
          "later than", "above", "over", "after", "exceeds", "exceeding", "exceed", "north of", "upwards of"),
    "<": ("less expensive than", "less than", "fewer than", "lower than", "smaller than", "cheaper than",
          "earlier than", "below", "under", "before", "sub"),
    "!=": ("not equal to", "not exactly", "different from"),
    "=": ("equal to", "equals", "exactly", "equal", "of", "at", "on"),
}
_CMP_MAP = {phrase: op for op, phrases in _CMP_PHRASES.items() for phrase in phrases}
_CMP_ALT = "|".join(re.escape(p) for p in sorted(_CMP_MAP, key=len, reverse=True))
_CMP_PREFIX = re.compile(rf"(?:\b(?P<word>{_CMP_ALT})\b|(?P<symbol>>=|<=|!=|>|<|=))\s*\$?\s*['\"]?$" )
_CMP_SUFFIX = re.compile(r"^['\"]?\s*(?:(?:units?|pieces?|dollars?|bucks?)\s+)?(?P<word>or more|or greater|or higher|or later"
                         r"|and above|and up|or above|or below|or less|or fewer|or earlier|or cheaper|or lower|and under|and below"
                         r"|up|tops|maximum|minimum|max|min)\b|^\s*(?P<plus>\+)")
_VALUE = r"(?:\d{4}-\d{2}-\d{2}|\d+(?:\.\d+)?)"
_VALUE_RE = re.compile(rf"(?<![\w.]){_VALUE}(?!\w|\.\d)")
_RANGE = re.compile(rf"\b(?:between|from)\s*\$?(?P<lo>{_VALUE})\s*(?:and|to|through|until|[-–])\s*\$?(?P<hi>{_VALUE})")
_COLUMN_WORD = re.compile(
    r"\b(?P<price>prices?|priced|pricing|costs?|costing|dollars?|bucks?)\b"
    r"|\b(?P<stock>stocks?|inventory|left|on hand|available)\b"
    r"|\b(?P<quantity>quantit(?:y|ies)|qty|units?|pieces?|order sizes?)\b"
    r"|\b(?P<order_date>order dates?|dates?|dated|placed|ordered)\b")
_FUNCTION_WORD = re.compile(r"\b(?P<AVG>average[sd]?|mean|avg)|\b(?P<SUM>total|sum|summed|combined)"
                            r"|\b(?P<MIN>minimum|lowest|smallest|earliest)|\b(?P<MAX>maximum|highest|largest|latest)"
                            r"|\b(?P<COUNT>count|counts|number of|how many)\b")
# 학습 조합기의 범주 명사도 개별 행을 가리킨다. 집계 키 명사(category 등)는 포함하지 않는다.
_ROW_NOUN = r"(?:items|products|things|orders|purchases|rows|clothes|clothing|sports (?:gear|equipment)|kitchen (?:stuff|items)|kitchenware)"


@dataclass(frozen=True)
class Comparison:
    """질문의 비교 위치. SQL 연산자에 의존하지 않고 값과 컬럼의 연결을 추출한다."""
    col: str | None
    op: str | None
    value: object
    value2: object = None
    start: int = 0
    end: int = 0
    aggregate: str | None = None


def _literal(text: str):
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else float(text)


def _near_column(q: str, start: int, end: int, table: str):
    """값에 가장 가까운 컬럼 단서를 읽고, 달러·날짜·단위 기본값을 적용한다."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", q[start:end]):
        return "order_date"
    if q[max(0, start - 2):start].strip().endswith("$"):
        return "price"
    left = q[max(0, start - 75):start]
    right = q[end:end + 32]
    before = list(_COLUMN_WORD.finditer(left))
    after = _COLUMN_WORD.search(right)
    if re.search(r"\bwith\s+(?:at least|at most|more than|less than)\s*$",left) and re.match(r"[^,;.]{0,32}\bin stock\b",right):
        return "stock"
    # 값 뒤의 in stock / units left 같은 표현은 앞의 조회 속성보다 우선한다.
    suffix = re.match(r"\s*(?:or more |or less |and above |and under )?(?:units? )?(?:in stock|left|on hand)\b", right)
    if suffix:
        return "stock"
    if re.match(r"\s+stock\b",right):
        return "stock"
    if re.match(r"\s*(?:(?:or more|or fewer|or less)\s+)?(?:units?|pieces?)\b", right):
        return "stock" if table == "items" else "quantity"
    if re.match(r"\s*(?:dollars?|bucks?)\b", right):
        return "price"
    if re.match(r"\s*(?:or more |or fewer |or less )?(?:orders|customers|items)\b", right):
        return "*"
    if re.search(r"\b(?:row count|order count|customer count|item count|count|counts|number of (?:customers|items|orders|rows)|total (?:orders|customers|items))\b[^,;?.]{0,30}$", left):
        return "*"
    if re.search(r"\b(?:maxima|minima|averages|means|totals|sums|maximum|minimum|average|mean|sum|total|that price)\b[^,;?.]{0,28}$", left):
        expressions = [(m.end(), col) for (fn, col), pattern in EXPRESSION.items()
                       for m in pattern.finditer(q[max(0, start - 220):start])]
        if expressions:
            alias = list(re.finditer(r"\b(?:maxima|minima|averages|means|totals|sums|maximum|minimum|average|mean|sum|total|that price)\b",left))
            if before and alias and before[-1].start() > alias[-1].end() and before[-1].lastgroup != max(expressions)[1]:
                return before[-1].lastgroup
            return max(expressions)[1]
    if before:
        last = before[-1]
        between = left[last.end():]
        if len(between) <= 35 and not re.search(r"[;?]|\b(?:items|products|things)\b", between):
            col = last.lastgroup
            return "stock" if col == "quantity" and table == "items" else col
    if after and after.start() < 18 and not re.search(r"[;,?]|\band\b|\bwith\b", right[:after.start()]):
        col = after.lastgroup
        return "stock" if col == "quantity" and table == "items" else col
    if re.search(r"\b(?:maxima|minima|averages|means|totals|sums)\b", left[-40:]):
        before = list(_COLUMN_WORD.finditer(left))
        if before:
            return before[-1].lastgroup
    return "price" if table == "items" else "quantity" if table == "orders" else None


@lru_cache(maxsize=32768)
def _comparisons(q: str, table: str) -> tuple[Comparison, ...]:
    """학습 문형의 접두·접미 비교와 범위를 값·컬럼에 연결한다."""
    result, covered = [], []
    for m in _RANGE.finditer(q):
        start, end = m.start("lo"), m.end("hi")
        result.append(Comparison(_near_column(q, start, m.end("lo"), table), "BETWEEN",
                                 _literal(m["lo"]), _literal(m["hi"]), start, end))
        covered.append((m.start(), m.end()))
    for m in _VALUE_RE.finditer(q):
        if any(lo <= m.start() < hi for lo, hi in covered):
            continue
        left, right = q[max(0, m.start() - 90):m.start()], q[m.end():]
        if re.search(r"\b(?:top|first|last|only|just)\s*$", left):
            continue
        prefix, suffix = _CMP_PREFIX.search(re.sub(r"\bboth\s*$", "", left)), _CMP_SUFFIX.search(right)
        op = _CMP_MAP[prefix["word"]] if prefix and prefix["word"] else prefix["symbol"] if prefix else None
        if (suffix and not re.match(r"\s+(?:the|that (?:(?:computed )?value|bound)|this|their|it|an?|overall|average|lowest|highest|minimum|maximum|expensive|costly|than|\$?\d+(?:\.\d+)?)\b", right[suffix.end():])
                and not (suffix["word"] in ("up", "and up") and re.match(r"\s+to\b", right[suffix.end():]))):
            op = ">=" if suffix["plus"] or suffix["word"] in (
                "or more", "or greater", "or higher", "or later", "and above", "and up", "up", "or above", "minimum", "min") else "<="
        # 제외한 행의 조건은 반대 경계다. 다른 AND 조건의 제외 단어를 빌리지 않는다.
        exclusion = re.search(r"\b(?:excluding|exclude|except|leaving out|omit|filtering out)\b([^,;.]{0,150})$", q[max(0,m.start()-180):m.start()])
        target = r"(?:items?|products?|orders?|prices?|priced|stock|quantit(?:y|ies)|totals?|sums?|counts?|groups?|cities|customers?|categor(?:y|ies)|minimums?|maximums?)"
        excluded_attribute = exclusion and (re.match(rf"\s+(?:any\s+)?{target}\b",exclusion[1])
            or re.match(rf"\s+['\"a-z -]+?\s+and\s+(?!priced\b){target}\b",exclusion[1]))
        excluded_tail = re.sub(rf"\b(?:{_CMP_ALT})\b", "", exclusion[1]) if exclusion else ""
        if (op in ("<", ">", "<=", ">=") and excluded_attribute
                and not re.search(r"\b(?:or|have|considering|using|keeping|requiring|only|then|return|show|report|give)\b|\band (?:with|priced)\b", excluded_tail)
                and not ("whose" in exclusion[1] and not re.search(r"\b(?:groups?|cities|categories|items?)\b",exclusion[1]))):
            op = {"<": ">=", ">": "<=", "<=": ">", ">=": "<"}[op]
        if re.search(r"\bno order quantity\b[^,;.]*$",left) or (exclusion and re.search(r"\bisn't\b",exclusion[1])):
            op = None  # 집합 부정·중첩 부정은 단순 비교 반전으로 확정하지 않는다.
        if op is None and re.search(r"\bfrom\s*$", left) and "-" in m.group():
            op = ">="
        result.append(Comparison(_near_column(q, m.start(), m.end(), table), op, _literal(m.group()),
                                 start=m.start(), end=m.end()))
    # 숫자가 생략되는 프로젝트의 고정 경계 표현도 비교식으로 읽는다.
    for pattern, op, val in ((r"\blow stock\b|\brunning low on stock\b", "<=", 10),
                             (r"\bin[- ]stock\b|\bpositive stock\b|\bavailable\b", ">", 0),
                             (r"\bout[- ]of[- ]stock\b|\bno stock\b|\bsold out\b", "=", 0)):
        for m in re.finditer(pattern, q):
            if op == "=" and re.search(r"\b(?:aren't|isn't|not)\s*$", q[max(0, m.start()-12):m.start()]):
                continue
            if val == 0 and op == ">" and re.search(r"\d\+?(?:\s+(?:or (?:more|fewer|less)|and (?:above|below)))?(?:\s+units?)?\s*$", q[max(0, m.start()-35):m.start()]):
                continue
            fixed_op = None if op == "=" and re.search(r"\b(?:exclude|excluding)\b[^,;.]*$",q[max(0,m.start()-60):m.start()]) else op
            result.append(Comparison("stock", fixed_op, val, start=m.start(), end=m.end()))
    return tuple(result)


@lru_cache(maxsize=3)
def _group_mentions_pattern(table: str):
    words = {key: "|".join(filter(None, (_GROUP_KEY_LABELS.get(key), GROUP_KEY_CUE[key])))
             for key in sql_ast.GROUP_COLUMNS[table]}
    keys = "|".join(rf"(?P<{key}>{value})" for key, value in words.items())
    return re.compile(rf"\b(?:per|for each|for every|each|by|sharing|according to)\s+(?:the\s+|each\s+|value of\s+)?(?:{keys})\b")


@lru_cache(maxsize=32768)
def _requested_groups(q: str, table: str):
    """그룹 전치사의 목적어를 읽되 정렬 구간·값이 붙은 필터·단가 표현은 제외한다."""
    found = []
    for m in _group_mentions_pattern(table).finditer(q):
        before, after = q[max(0, m.start()-35):m.start()], q[m.end():]
        if m.group().startswith("by ") and re.search(r"\b(?:sort(?:ed|ing)?|order(?:ed|ing)?|rank(?:ed|ing)?)(?: groups)?\s*$", before):
            continue
        if re.match(r"\s*(?:ids?|numbers?|no\.?)?\s*#?\s*\d", after):
            continue
        found.append((m.lastgroup, m.group()))
    for key in sql_ast.GROUP_COLUMNS[table]:
        words = "|".join(filter(None, (_GROUP_KEY_LABELS.get(key), GROUP_KEY_CUE[key])))
        for m in re.finditer(rf"\b(?:{words})\s+(?:breakdown|groups?|grouping)\b", q):
            if re.search(rf"\b(?:{_AGG_WORD}|latest|earliest)\s+(?:order\s+)?$", q[max(0,m.start()-25):m.start()]):
                continue
            found.append((key, m.group()))
        for pattern in (rf"\b(?:{words})\s+as (?:the )?(?:key|grouping attribute)\b",
                        rf"\b(?:every represented|corresponding)\s+(?:{words})\s+value\b"):
            found.extend((key, m.group()) for m in re.finditer(pattern,q))
    return found


@lru_cache(maxsize=32768)
def _comparison_scope(q: str, col: str | None, start: int) -> str | None:
    """집계 표현과 개별 행 명사 중 비교값에 실제로 이어지는 구간을 구분한다."""
    left = q[max(0, start - 180):start]
    if col == "stock" and re.match(rf"(?:in[- ]stock|available)\s+{_ROW_NOUN}\b",q[start:]):
        return "where"
    # 집계 대상의 수식인지 집계 결과의 술어인지 연결어 없이 붙은 두 학습 문형은 확정하지 않는다.
    if re.search(rf"\b(?:its (?:smallest|largest|minimum|maximum) (?:order )?quantity|total item prices)\s+(?:{_CMP_ALT})\s*$",left):
        return None
    expressions = [(m.end(), m.start()) for (fn, c), pattern in EXPRESSION.items()
                   if c == col for m in pattern.finditer(left)]
    if col == "*":
        expressions.extend((m.end(), m.start()) for m in re.finditer(
            r"\b(?:count|counts|number of|total (?:orders|customers|items)|tally)\b", left))
    # 앞 집계를 다시 지칭하는 대명사형: keeping averages / totals / counts above ...
    expressions.extend((m.end(), m.start()) for m in re.finditer(
        rf"\b(?:averages|means|sums|totals|maxima|minima|counts|minimums|maximums|averaging|sum|total|average|mean|minimum|maximum|lowest|highest|smallest|largest|cheapest|peak|biggest|totaling|add up|sum to|that price|keep results)\b", left)
                       if not re.search(r"\b(?:at|no|not)\s+$", left[:m.start()]))
    expression = max(expressions, default=None)
    rows = list(re.finditer(r"\b(?:items|products|things|orders|purchases|customers|clients)\b", left))
    last_row = rows[-1].end() if rows else -1
    if re.search(r"\bprovided it\s+(?:is\s+)?[^,;.]*$",left):
        return "having"
    if col == "quantity" and re.search(r"\bdaily ordered units\b[^,;.]*$",left):
        return "having"
    if col == "quantity" and re.search(rf"\beach order date\b[^,;.]*\bwhere\s+(?:{_CMP_ALT})\s*$",left):
        return "having"
    if col == "*" and re.search(r"\bcustomer ids? who placed\s+[^,;.]*$",left):
        return "having"
    if col == "*" and re.search(r"\b(?:each|per|by)\b",left) and re.search(r"\b(?:with|have|having)\s+[^,;.]{0,30}$",left):
        return "having"
    if col == "quantity" and re.search(r"\b(?:customers?|customer ids?|item ids?|items?|statuses|dates?)\s+(?:with|have|whose|ordered|reaching)\s+[^,;.]{0,35}$",left) and not re.search(r"\b(?:orders|purchases)\b[^,;.]{0,30}$",left):
        return "having"
    if expression:
        # number of orders는 집계 안의 대상 명사이며 새 행 필터의 시작이 아니다.
        tail = left[expression[0]:]
        if col == "order_date" or re.search(r"\bqualifying\s+(?:price|stock|quantity)\b",left[expression[1]:]):
            return None  # 집계 대상 제한과 집계 결과 제한이 문장만으로 구별되지 않는다.
        if col != "*" and re.search(rf"\b{_ROW_NOUN}\b.*\b(?:stock|prices?|priced|quantity|quantities)\b",tail):
            return "where"
        if re.search(r"\b(?:for|from) (?:stock|prices?|quantity|quantities)\b",tail):
            return "where"
        if re.search(r"\b(?:is|are)\s+(?:at least|at most|less than|more than|under|above|below|over|>=|<=)?\s*$",tail):
            return "having"
        if re.search(r"\b(?:filtering|restricted|limited|using|considering|excluding)\s+(?:only\s+|to\s+)?(?:prices?|quantit(?:y|ies)|stock)\b", tail) or re.search(r"\bafter excluding\b",tail):
            return "where"
        if col != "*" and (re.search(rf"\b{_ROW_NOUN}\b", tail)
                           or re.search(r"\b(?:considering|using|include|including)\s+(?:only\s+)?(?:quantities|prices|stock)\b", tail)):
            return "where"
        if re.search(r"\b(?:whose|where|provided|only when|only where)\b[^,;.]{0,45}$", tail) and not re.search(r"\b(?:priced|stock|quantity|quantities)\b", tail):
            return "having"
        if not re.search(r"\b(?:items|products|things|orders|purchases|customers|clients)\b.*"
                         r"\b(?:priced|costing|with|whose|having|stock|quantity|dated)\b", tail):
            if len(tail) < 85 and not re.search(r"[.;?]", tail):
                return "having"
    if col == "*" and re.search(r"\b(?:customers?|items?|categories|cities|statuses|dates?|tiers?|memberships?|levels?|groups?)\s+(?:group\s+)?(?:with|having|have)\s+[^,;.]{0,35}$", left):
        return "having"
    if last_row >= 0 and not expression:
        return "where"
    if not expression and col and re.search(r"\b(?:stock|quantity|price|priced|costing|units|dated|date)\b[^,;.]{0,40}$", left):
        return "where"
    return None


def _where_having_judgment(q: str, g) -> Judgment | None:
    if not isinstance(g, sql_ast.AdvancedQuery) or not g.group_by or g.subquery:
        return None
    aggregate_col = g.aggregate[:-1].split("(")[1]
    unchecked = []
    for c in _comparisons(q, g.table):
        if c.col != aggregate_col or c.op is None or c.op == "BETWEEN":
            continue
        scope = _comparison_scope(q, c.col, c.start)
        row = any(cond.col == c.col and c.value in _cond_values(cond) for cond in g.conds)
        # IN의 개별 값은 임계값이 아니다. 같은 값이 HAVING에도 나올 수 있다.
        if any(cond.col == c.col and cond.op == "IN" and c.value in cond.val for cond in g.conds) and scope == "where":
            continue
        having = bool(g.having and g.having[1] == c.value)
        if (scope == "where" and having and not row) or (scope == "having" and row and not having):
            return Judgment("violation", "L22", [f"{q[max(0,c.start-55):c.end]}: 질문 {scope}, SQL은 반대 범위"])
        if scope is None and (row or having):
            unchecked.append(f"WHERE/HAVING 연결: {c.col} {c.value}")
    return Judgment("review", unchecked=unchecked) if unchecked else None


def _group_judgment(q: str, g) -> Judgment | None:
    if not isinstance(g, sql_ast.AdvancedQuery) or g.subquery:
        return None
    groups = _requested_groups(q, g.table)
    keys = {key for key, _ in groups}
    if len(keys) == 1 and g.group_by not in keys:
        return Judgment("violation", "L21", [f"그룹 요청: {text}" for _, text in groups] + [f"SQL 그룹: {g.group_by}"])
    if len(keys) > 1:
        return Judgment("review", evidence=[text for _, text in groups], unchecked=["복수 그룹 요청의 연결 관계"])
    return None


def _comparison_judgment(q: str, g) -> Judgment | None:
    observations = _comparisons(q, g.table)
    conditions = list(g.conds) + list(getattr(g, "inner_conds", ()))
    checks = [(c.col, c.op, c.val, "inner" if i >= len(g.conds) else "outer")
              for i, c in enumerate(conditions) if c.op in ("<", ">", "<=", ">=")]
    if getattr(g, "having", None):
        checks.append((g.aggregate[:-1].split("(")[1], g.having[0], g.having[1], "having"))
    unchecked = []
    reference = _reference_span(q) if getattr(g,"subquery",None) else None
    for col, op, val, scope in checks:
        matches = [c for c in observations if c.value == val and c.col == col and c.op != "BETWEEN"]
        if reference and not BACK_REFERENCE.search(q):
            scoped = [c for c in matches if ("inner" if reference[0] <= c.start < reference[1] else "outer") == scope]
            if scoped:
                matches = scoped
        if isinstance(g, sql_ast.AdvancedQuery) and g.group_by and not g.subquery:
            wanted = "having" if scope == "having" else "where"
            scoped = [c for c in matches if _comparison_scope(q,c.col,c.start) == wanted]
            if scoped:
                matches = scoped
        known = {c.op for c in matches if c.op is not None}
        if known and op not in known and all(c.op is not None for c in matches):
            return Judgment("violation", "L22", [f"{col} {val}: 질문 {sorted(known)}, SQL {op}"])
        if matches and (not known or any(c.op is None for c in matches)):
            unchecked.append(f"비교 방향: {col} {val}")
    if getattr(g, "subquery", None):
        # 집계 기준값 바로 앞 비교만 읽는다. 안쪽 행 필터의 비교 단어와 구분한다.
        pattern = rf"\b(?P<cmp>{_CMP_ALT})\s+(?:(?:the|overall|global|all[- ]item|all[- ]order)\s+)*(?:{_AGG_WORD}|computed value|it|that (?:computed )?value|the result|that bound)\b"
        # min / max 뒤 숫자는 행 조건의 경계이며 집계 기준값이 아니다.
        # 기준 구간 추출과 동일하게 숫자 경계 후보를 건너뛴다.
        match = next((m for m in re.finditer(pattern, q)
                      if not re.match(r"\s*\$?\d",q[m.end():])), None)
        if match:
            expected = _CMP_MAP[match["cmp"]]
            prefix = q[max(0,match.start()-45):match.start()]
            if expected == "=" and re.search(r"\bexcluding\s+(?:any\s+)?(?:prices?|priced|items?\s+priced)\s*$",prefix):
                expected = "<" if g.subquery[2].startswith("MAX(") else ">" if g.subquery[2].startswith("MIN(") else "!="
            if re.search(r"\bmeet it or exceed it\b|\bas much as or more than\b",q):
                expected = ">="
            if expected != g.subquery[1]:
                return Judgment("violation", "L23", [f"서브쿼리 비교: 질문 {expected}, SQL {g.subquery[1]}"])
        if not match:
            unchecked.append("서브쿼리 기준값에 붙은 비교 방향")
    return Judgment("review", unchecked=unchecked) if unchecked else None


@lru_cache(maxsize=32768)
def _reference_span(q: str) -> tuple[int, int] | None:
    """기준 집계부터 조회 요청이 다시 시작되는 지점까지를 기준 집단 구간으로 읽는다."""
    match = next((m for m in re.finditer(rf"\b{_AGG_WORD}\b", q)
                  if not re.match(r"\s*\$?\d", q[m.end():])
                  and not (m.group() in ("max", "min", "maximum", "minimum")
                           and re.search(r"\d(?:\s+(?:dollars?|bucks?|units?))?\s*$",q[max(0,m.start()-25):m.start()]))), None)
    if not match:
        return None
    # 소수점·날짜의 기호는 문장 경계가 아니다.
    start = match.start()
    labels = "|".join(re.escape(v) for values in CATEGORICAL_VALUES.values() for v in values)
    modifier = re.search(rf"\b(?:non[- ])?['\"]?(?:{labels})['\"]?(?:[- ](?:category|order|item)| (?:category|orders?))?(?:'s|')?\s+$", q[:start])
    if modifier:
        start = modifier.start()
    population = re.search(r"\b(?:use|using)\s+(?:the\s+)?", q[:start])
    if population and (re.search(r"\bto (?:compute|calculate|establish|get|find|determine)\s+(?:the\s+)?$|\b(?:for|as) (?:the\s+)?$", q[:start])
                       or re.search(r"\bto work out\s+(?:the\s+)?$",q[:start])
                       or "reference population" in q[population.end():start]):
        start = population.start()
    tail = q[start:]
    boundary = re.search(r"[.;?](?=\s+[a-z])|\b(?:and\s+)?(?:return|report|list|show|provide|retrieve|display|give|can i get|requested output|send me|grab|fetch|get (?!the\b)|find (?:names|item names|orders)|select|enumerate|include (?:item names|all details|order ids))\b"
                         r"|\b(?:and|then) use it to (?:find|fetch|get|select)\b", tail)
    end = start + boundary.start() if boundary else len(q)
    outer_tail = re.search(r",\s*(?:restricted to|limited to|provided|keeping|excluding those|but only|fall within|for (?:the )?(?:items|orders|products)(?!\s+#?\d))\b|\band (?:has|belong to|is at|satisfying|no more than|at most)(?!\s*\$)\b", q[match.end():end])
    if not outer_tail and not re.search(r"\bfrom\s+\d",q[match.end():end]):
        outer_tail = re.search(r"\bthrough\s+\d+(?![\d-])",q[match.end():end])
    if outer_tail:
        end = match.end() + outer_tail.start()
    continuation = q[match.end():end]
    if (re.search(r"\b(?:overall|global|catalog-wide|all[- ]order|all[- ]item)\s+(?:[a-z]+\s+){0,2}$",q[max(0,start-35):start])
            or re.search(r"\b(?:across|among|of|for) all (?:items|orders)\b(?!\s+(?:placed|dated|with|priced))",continuation)):
        extra = re.search(r"\band\b|\bwhile remaining\b",continuation)
        if extra:
            end = match.end()+extra.start()
    if re.search(r"\b(?:overall|global|all[- ]order|all[- ]item)\s+(?:[a-z]+\s+){0,2}$",q[max(0,start-30):start]):
        extra = re.search(r"\b(?:for|with|through)\b",q[match.end():end])
        if extra:
            end = match.end()+extra.start()
    return start, end


@lru_cache(maxsize=32768)
def _condition_positions(q: str, table: str, cond):
    """조건 리터럴의 위치를 모은다. 이름은 마스킹 전 원문에서 확인한다."""
    positions = []
    missing = []
    for value in _cond_values(cond):
        if isinstance(value, str):
            text = value.strip("%").lower()
            variants = [re.escape(text)]
            if text == "clothing":
                variants.append("clothes")
            if text == "office supplies":
                variants.append("office supply")
            if text == "kitchen":
                variants.extend(("kitchenware", "kitchen stuff"))
            if text == "pending":
                variants.append("waiting to ship")
            pattern = rf"(?<![a-z0-9])(?:{'|'.join(variants)})(?![a-z0-9])"
        else:
            number = str(float(value)).rstrip("0").rstrip(".") if float(value) % 1 else str(int(value))
            zeros = "0*" if "." in number else r"(?:\.0+)?"
            pattern = rf"(?<![\w.]){re.escape(number)}{zeros}(?![\w]|\.\d)"
        found = [m.start() for m in re.finditer(pattern, q)]
        if found:
            positions.extend(found)
        else:
            missing.append(str(value))
    return positions, missing


def _subquery_scope_judgment(q: str, g) -> Judgment | None:
    if not isinstance(g, sql_ast.AdvancedQuery) or not g.subquery:
        return None
    span = _reference_span(q)
    if span is None:
        return Judgment("review", unchecked=["서브쿼리 기준 집단 구간"])
    start, end = span
    back = BACK_REFERENCE.search(q)
    unchecked = []
    for scope, conditions in (("outer", g.conds), ("inner", g.inner_conds)):
        for cond in conditions:
            positions, missing = _condition_positions(q, g.table, cond)
            observed = {"inner" if start <= pos < end else "outer" for pos in positions}
            if cond.col == "category" and cond.op == "!=" and "other categories" in q[start:end]:
                observed.add("inner")
            if back and "outer" in observed:
                observed.add("inner")
            if observed and scope not in observed and not missing:
                # 집계 뒤 쉼표의 for items / with order ids는 두 집합 모두를 수식할 수 있다.
                # return 등 명시적인 조회 재개와 달리 어느 쪽이라고 확정하지 않는다.
                if re.match(r",\s*(?:for (?:the )?(?:items|products|orders)|with order ids?)\b",q[end:]):
                    unchecked.append(f"서브쿼리 뒤 꼬리 조건의 수식 대상: {cond.col} {cond.val}")
                    continue
                if scope == "outer" and re.search(r"\band (?:whose|prices?|stock|quantit(?:y|ies))\b|, with quantity\b",q[start:end]):
                    unchecked.append(f"서브쿼리 뒤 병렬 조건의 수식 대상: {cond.col} {cond.val}")
                    continue
                return Judgment("violation", "L23", [f"기준 구간: {q[start:end]}",
                    f"{cond.col} {cond.val}: 질문 {sorted(observed)}, SQL {scope}"])
            if missing:
                segment = q[start:end] if scope == "inner" else q[:start] + " " + q[end:]
                if cond.col in ("category", "status") and cond.op == "=" and any(
                        re.search(rf"\b{re.escape(v)}\b",segment) for v in CATEGORICAL_VALUES[g.table]):
                    return Judgment("violation", "L23", [f"{scope}의 명시된 범주와 SQL {cond.col}={cond.val} 불일치"])
                unchecked.append(f"기준 집단 값 연결: {cond.col} {', '.join(missing)}")
    return Judgment("review", unchecked=unchecked) if unchecked else None


def judge(pair: dict) -> Judgment:
    """확정 위반만 reason을 반환하고 실제로 해석하지 못한 표현은 검토 대상으로 남긴다."""
    try:
        g = _parse_query(pair["sql"])
    except (ValueError, KeyError, TypeError):
        return Judgment("review", unchecked=["지원 문법 밖 SQL의 질문 의미"])
    q = pair["question"].lower().strip().replace("_", " ")
    reason = _legacy_reason(pair, g)
    if reason:
        evidence = [f"{reason}: {REASONS[reason]}"]
        evidence.extend(f"질문 단서: {m.group()}" for p in FUNCTION_CUE.values() if (m := p.search(q)))
        return Judgment("violation", reason, evidence)
    if re.search(r"\b(?:in the vicinity of|roughly|approximately|around|about)\s+\$?\d", q):
        return Judgment("review", evidence=["근사 수치 표현"], unchecked=["비교 경계·방향"])
    masked = _mask_names(q, g) if isinstance(g, sql_ast.AdvancedQuery) else q
    group = _group_judgment(masked, g)
    comparison = _comparison_judgment(masked, g)
    scope = _where_having_judgment(masked, g)
    subquery = _subquery_scope_judgment(q, g)
    if group and group.status == "violation":
        return group
    if comparison and comparison.status == "violation":
        return comparison
    if scope and scope.status == "violation":
        return scope
    if subquery and subquery.status == "violation":
        return subquery
    reviews = [j for j in (group, comparison, scope, subquery) if j and j.status == "review"]
    return (Judgment("review", evidence=[e for j in reviews for e in j.evidence],
                     unchecked=list(dict.fromkeys(e for j in reviews for e in j.unchecked)))
            if reviews else Judgment("pass"))


def drop_reason(pair: dict) -> str | None:
    """기존 호출자에는 확정 위반만 노출한다. 검토 필요를 오답으로 처리하지 않는다."""
    result = judge(pair)
    return result.reason if result.status == "violation" else None
