import pytest

from src.data.stage1 import sql_rules


# ===========================================================================
# parse_sql / error_parts
# ===========================================================================

@pytest.mark.unit
def test_parse_sql_stage1():
    """1단계 SQL 을 select / table / where_col / literal 로 분해하는지 검증"""
    assert sql_rules.parse_sql("SELECT city FROM customers WHERE name = 'ashley'") == {
        "select": "city", "table": "customers", "where_col": "name", "literal": "'ashley'"}
    assert sql_rules.parse_sql("SELECT * FROM items WHERE price = 14.46")["select"] == "*"


@pytest.mark.unit
@pytest.mark.parametrize("sql", [None, "", "SELECT city FROM customers", "DROP TABLE items"])
def test_parse_sql_rejects_non_stage1(sql):
    """1단계 문법이 아니거나 None 이면 None"""
    assert sql_rules.parse_sql(sql) is None


@pytest.mark.unit
@pytest.mark.parametrize("pred, expected", [
    ("SELECT city FROM customers WHERE name = 'ashley'", []),
    ("SELECT city FROM customers WHERE name = 'ashly'", ["literal"]),
    ("SELECT * FROM customers WHERE name = 'ashley'", ["select"]),
    ("SELECT city FROM orders WHERE customer_id = 'ashley'", ["table", "where_col"]),
    (None, ["malformed"]),
    ("SELECT city FROM", ["malformed"]),
])
def test_error_parts(pred, expected):
    """정답과 다른 부분(table / where_col / select / literal)을 정확히 분류하는지 검증"""
    assert sql_rules.error_parts("SELECT city FROM customers WHERE name = 'ashley'", pred) == expected


# ===========================================================================
# select_ambiguous / multi_match (다중 정답 EM)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("question, gold, expected", [
    # 개체를 묻지만 이름인지 행 전체인지 단서가 없음
    ("which customer lives in paris?", "SELECT * FROM customers WHERE city = 'paris'", True),
    # "all" → 행 전체 단서
    ("show all details of customers in paris", "SELECT * FROM customers WHERE city = 'paris'", False),
    # "name" → 식별 컬럼 단서
    ("what's the name of the customer in paris?", "SELECT name FROM customers WHERE city = 'paris'", False),
    # SELECT 가 * / 식별 컬럼이 아니면 대상 아님
    ("what city is ashley in?", "SELECT city FROM customers WHERE name = 'ashley'", False),
    # 존재 확인형 (SELECT == WHERE 컬럼) 은 제외
    ("is there a customer named ashley?", "SELECT name FROM customers WHERE name = 'ashley'", False),
    # WHERE 조건 표현("named ashley")은 SELECT 단서로 보지 않음
    ("pull up the customer named ashley", "SELECT * FROM customers WHERE name = 'ashley'", True),
    # orders: "item id 3155" 는 WHERE 조건 표현이라 ID 단서가 아님
    ("orders for item id 3155", "SELECT * FROM orders WHERE item_id = 3155", True),
    ("order numbers for item 3155", "SELECT order_id FROM orders WHERE item_id = 3155", False),
    ("anything", "not sql", False),
])
def test_select_ambiguous(question, gold, expected):
    """SELECT * ↔ 식별 컬럼 모호 문항 판정 규칙 검증"""
    assert sql_rules.select_ambiguous(question, gold) is expected


@pytest.mark.unit
def test_multi_match():
    """모호 문항에서는 SELECT 만 * ↔ 식별 컬럼으로 다른 예측도 정답, 그 외는 오답"""
    q, gold = "which customer lives in paris?", "SELECT * FROM customers WHERE city = 'paris'"
    assert sql_rules.multi_match(q, gold, gold)
    assert sql_rules.multi_match(q, gold, "SELECT name FROM customers WHERE city = 'paris'")
    assert not sql_rules.multi_match(q, gold, "SELECT city FROM customers WHERE city = 'paris'")
    assert not sql_rules.multi_match(q, gold, "SELECT name FROM customers WHERE city = 'rome'")
    assert not sql_rules.multi_match(q, gold, None)
    # 단서가 있는 문항은 완전 일치만 정답
    q2 = "show all details of customers in paris"
    assert not sql_rules.multi_match(q2, gold, "SELECT name FROM customers WHERE city = 'paris'")
