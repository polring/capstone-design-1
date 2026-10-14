import pytest

from src.data.stage2.clean_ambiguous import drop_reason

pytestmark = pytest.mark.unit


def r(question, sql):
    return drop_reason({"question": question, "sql": sql})


def test_l1_existence_needs_cue():
    """L1: SELECT 가 = 조건 컬럼이면 존재 확인 질문이어야 한다 (1단계 단서 + do we have, 활용형)"""
    sql = "SELECT price FROM items WHERE price = 9.66"
    assert r("is there an item priced 9.66?", sql) is None
    assert r("do you have anything that costs exactly $9.66", sql) is None
    assert r("would you mind checking if there's an item priced 9.66", sql) is None
    assert r("what is the price of the item priced 9.66?", sql) == "L1"
    assert r("do we even have an item priced 9.66?", sql) is None  # 부사가 낀 do we have (LLM 학습 질문)


def test_interjection_before_question_word():
    """문장 첫머리 감탄사 뒤의 which / who 도 의문문이다 ("hey, who're …", LLM 학습 질문)"""
    sql = "SELECT name FROM customers WHERE membership = 'silver'"
    assert r("hey, who're the silver members?", sql) is None
    assert r("um, who are the silver members?", sql) is None
    assert r("show me the silver members", sql) == "L2"


def test_l2_star_or_ident_without_cue():
    """L2: 단서가 없으면 ① 식별 컬럼 조건 → *, ② which / who 의문문 → 식별, ③ 복수 → *, ④ 조회 동사 → *, ⑤ 식별"""
    oid = "SELECT order_id FROM orders WHERE item_id = 3543"
    star = "SELECT * FROM orders WHERE item_id = 3543"
    assert r("can you list the orders for item 3543?", oid) == "L2"
    assert r("can you list the orders for item 3543?", star) is None
    assert r("which orders are for item 3543?", oid) is None
    assert r("look up which order has item 3543", oid) is None
    assert r("what item costs 91.52?", "SELECT item_name FROM items WHERE price = 91.52") is None
    # 관계대명사 who 는 의문문이 아니다 (조합기 star 틀 "show me the customers … who live in denver")
    assert r("show me the customers who live in denver", "SELECT * FROM customers WHERE city = 'denver'") is None
    assert r("find the item that costs 76.40", "SELECT item_name FROM items WHERE price = 76.40") == "L2"  # L13: find = star


def test_l3_own_id_needs_id_cue():
    """L3: 자기 ID 는 질문이 ID 를 요구할 때만. ID 조건 값 앞의 'id' 는 단서가 아니다"""
    assert r("which item has 94 left?", "SELECT item_id FROM items WHERE stock = 94") == "L3"
    assert r("item id of the thing with 94 left", "SELECT item_id FROM items WHERE stock = 94") is None
    assert r("item id 174.13", "SELECT item_id FROM items WHERE price = 174.13") is None


def test_l4_orders_table_needs_order_cue():
    """L4: customers 로도 성립하는 orders SQL 은 주문 단서가 있어야 한다"""
    assert r("customer 1400 details", "SELECT * FROM orders WHERE customer_id = 1400") == "L4"
    assert r("orders for customer 1400", "SELECT * FROM orders WHERE customer_id = 1400") is None


def test_l14_existence_selects_ident():
    """L14: 조건 여러 개의 존재 확인은 식별 컬럼. 행 전체 단서가 있으면 조회(*)다"""
    sql = "SELECT * FROM items WHERE category = 'furniture' AND price < 40.00"
    assert r("do we have any furniture under $40?", sql) == "L14"
    assert r("do we have any furniture under $40?", sql.replace("SELECT *", "SELECT item_name")) is None
    assert r("is there any info on furniture under $40?", sql) is None
