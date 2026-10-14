"""
2단계 스키마 메타데이터: 컬럼 종류, 허용 연산자, 정렬 가능 여부, 리터럴 표기.

DB 자체(shop.db, holdout.json)는 1단계 db_gen 것을 그대로 쓴다. 여기서는 2단계 SQL 문법이
컬럼마다 무엇을 허용하는지만 정한다.
"""

from __future__ import annotations

from src.data import db_gen

TABLE_COLUMNS: dict[str, list[str]] = db_gen.TABLE_COLUMNS

# 컬럼 종류
#   id   : ID (PK·FK)         -> =
#   name : 한 단어 이름        -> =, LIKE
#   cat  : 범주형 텍스트       -> =, !=
#   num  : 숫자                -> =, <, >, <=, >=, BETWEEN  (ORDER BY 가능)
KIND: dict[tuple[str, str], str] = {
    ("customers", "customer_id"): "id",
    ("customers", "name"): "name",
    ("customers", "city"): "cat",
    ("customers", "membership"): "cat",
    ("items", "item_id"): "id",
    ("items", "item_name"): "name",
    ("items", "category"): "cat",
    ("items", "price"): "num",
    ("items", "stock"): "num",
    ("orders", "order_id"): "id",
    ("orders", "customer_id"): "id",
    ("orders", "item_id"): "id",
    ("orders", "quantity"): "num",
    ("orders", "status"): "cat",
}

OPS: dict[str, tuple[str, ...]] = {
    "id": ("=",),
    "name": ("=", "LIKE"),
    "cat": ("=", "!="),
    "num": ("=", "<", ">", "<=", ">=", "BETWEEN"),
}
ALL_OPS = ("=", "!=", "<", ">", "<=", ">=", "BETWEEN", "LIKE", "IN")
IN_MAX = 3  # IN 값 개수 상한 (2 ~ IN_MAX)

# 실수 컬럼. 리터럴은 값과 무관하게 소수 둘째 자리로 쓴다 (50 -> 50.00)
FLOAT_COLUMNS = {("items", "price")}

# 행 하나를 가리키는 키. 이 컬럼에 = 조건이 걸리면 결과가 최대 1행이라 다른 AND 조건·ORDER BY 를 붙이지 않는다
UNIQUE_KEYS = {("customers", "customer_id"), ("customers", "name"),
               ("items", "item_id"), ("items", "item_name"), ("orders", "order_id")}

# 식별 컬럼 / 자기 ID (라벨 규칙 용어, docs/text-to-sql-label-rules.md)
IDENT = {"customers": "name", "items": "item_name", "orders": "order_id"}
OWN_ID = {"customers": "customer_id", "items": "item_id"}


def kind(table: str, col: str) -> str:
    return KIND[(table, col)]


def in_ok(table: str, col: str) -> bool:
    """같은 컬럼 값 나열(IN)을 쓸 수 있는가: = 가 되는 컬럼 중 price 는 뺀다 (가격 여러 개를 나열하는 질문은 부자연스러움)."""
    return "=" in OPS[kind(table, col)] and (table, col) not in FLOAT_COLUMNS


def ops(table: str, col: str) -> tuple[str, ...]:
    """그 컬럼에 쓸 수 있는 연산자 (종류별 OPS + IN)."""
    return OPS[kind(table, col)] + (("IN",) if in_ok(table, col) else ())


def orderable(table: str, col: str) -> bool:
    return KIND.get((table, col)) == "num"


def is_text(table: str, col: str) -> bool:
    return KIND[(table, col)] in ("name", "cat")


def col_index(table: str, col: str) -> int:
    return TABLE_COLUMNS[table].index(col)


def fmt_literal(table: str, col: str, val) -> str:
    if is_text(table, col):
        return f"'{val}'"
    if (table, col) in FLOAT_COLUMNS:
        return f"{float(val):.2f}"
    return str(int(val))


def select_kind(table: str, select: str, cond_cols: set[str]) -> str:
    """SELECT 종류: * / ident(식별 컬럼) / own_id(자기 ID) / cond_col(조건 컬럼과 같음) / attr(그 외 속성)."""
    if select == "*":
        return "*"
    if select in cond_cols:
        return "cond_col"
    if select == IDENT[table]:
        return "ident"
    if select == OWN_ID.get(table):
        return "own_id"
    return "attr"
