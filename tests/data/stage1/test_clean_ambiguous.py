import pytest

from src.data.stage1.clean_ambiguous import drop_reason, select_label
from src.data.stage1.sql_rules import parse_sql


# ===========================================================================
# select_label (L2)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("question, sql, expected", [
    # WHERE 컬럼이 식별 컬럼이면 SELECT name 은 존재 확인형이 되므로 *
    ("pull up ashley", "SELECT * FROM customers WHERE name = 'ashley'", "*"),
    ("which customer lives in paris?", "SELECT * FROM customers WHERE city = 'paris'", "name"),
    ("who lives in paris?", "SELECT * FROM customers WHERE city = 'paris'", "name"),
    ("customers in paris", "SELECT * FROM customers WHERE city = 'paris'", "*"),
    ("show me the gold member in paris", "SELECT * FROM customers WHERE city = 'paris'", "*"),
    ("the gold member in paris", "SELECT * FROM customers WHERE city = 'paris'", "name"),
    ("which order is pending?", "SELECT * FROM orders WHERE status = 'pending'", "order_id"),
])
def test_select_label(question, sql, expected):
    """L2 규칙 순서(WHERE=식별 컬럼 → which/who → 복수형 → 행 동사 → 식별 컬럼) 검증"""
    assert select_label(question, parse_sql(sql)) == expected


# ===========================================================================
# drop_reason
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("question, sql, expected", [
    # L2: 모호 문항인데 규칙과 다른 SELECT
    ("which customer lives in paris?", "SELECT * FROM customers WHERE city = 'paris'", "select"),
    ("which customer lives in paris?", "SELECT name FROM customers WHERE city = 'paris'", None),
    # L3: 질문이 ID 를 요구하지 않는데 자기 ID 를 SELECT
    ("what item costs 14.46?", "SELECT item_id FROM items WHERE price = 14.46", "id"),
    ("what's the item id of the thing priced 14.46?", "SELECT item_id FROM items WHERE price = 14.46", None),
    ("customer # for whoever lives in paris", "SELECT customer_id FROM customers WHERE city = 'paris'", None),
    # L3 예외: SELECT == WHERE 컬럼 (존재 확인형)
    ("is there an item 3155?", "SELECT item_id FROM items WHERE item_id = 3155", None),
    # L4: orders 인데 주문 단서가 없음
    ("show me customer 1400", "SELECT * FROM orders WHERE customer_id = 1400", "table"),
    ("show me the orders for customer 1400", "SELECT * FROM orders WHERE customer_id = 1400", None),
    ("is customer 1400 in there?", "SELECT customer_id FROM orders WHERE customer_id = 1400", "table"),
    ("anything purchased by customer 1400?", "SELECT customer_id FROM orders WHERE customer_id = 1400", None),
    # 일반 문항 / 1단계 문법 밖
    ("what city does ashley live in?", "SELECT city FROM customers WHERE name = 'ashley'", None),
    ("whatever", "SELECT city FROM customers", None),
])
def test_drop_reason(question, sql, expected):
    """라벨 규칙 L2/L3/L4 위반 사유 판정 검증"""
    assert drop_reason({"question": question, "sql": sql}) == expected


@pytest.mark.unit
def test_drop_reason_is_case_insensitive():
    """질문 대소문자와 무관하게 같은 판정"""
    sql = "SELECT * FROM customers WHERE city = 'paris'"
    assert drop_reason({"question": "WHICH customer lives in paris?", "sql": sql}) == "select"
