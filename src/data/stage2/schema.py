# 역할: 2단계 컬럼 종류, 연산자와 값 표기 규칙을 정의한다.
# 사용처: 2단계 SQL 구조·생성기·질문 조합기가 문법과 컬럼별 처리를 결정할 때 사용한다.

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


def kind(table, col):
    return KIND[table, col]


def in_ok(table, col):
    return kind(table, col) != 'date' and '=' in OPS[kind(table, col)] and ((table, col) not in FLOAT_COLUMNS)


def ops(table, col):
    return OPS[kind(table, col)] + (('IN',) if in_ok(table, col) else ())


def orderable(table, col):
    return KIND.get((table, col)) in ('num', 'date')


def is_text(table, col):
    return kind(table, col) in ('name', 'cat', 'date')


def col_index(table, col):
    return TABLE_COLUMNS[table].index(col)


def fmt_literal(table, col, val):
    if is_text(table, col):
        return f"'{val}'"
    return f'{float(val):.2f}' if (table, col) in FLOAT_COLUMNS else str(int(val))


def select_kind(table, select, cond_cols):
    if select == '*':
        return '*'
    if select in cond_cols:
        return 'cond_col'
    if select == IDENT[table]:
        return 'ident'
    if select == OWN_ID.get(table):
        return 'own_id'
    return 'attr'
