"""3단계 라벨 검사기 보강 목표(A2~A6). 짝마다 "잡아야 할 틀린 라벨"과 "통과해야 할 정상 라벨"을 둔다.

아직 구현하지 않은 판정은 xfail(strict=True)이다. 구현해서 통과하면 strict 때문에 실패로 바뀌므로 그 표시를 지운다.
정상 라벨 테스트(guard)는 지금도 통과하며, 보강 뒤에도 계속 통과해야 한다(오탐 방지).
"""
import pytest

from src.data.stage3 import clean_ambiguous as ca
from src.data.stage3.clean_ambiguous import drop_reason

pytestmark = pytest.mark.unit
TODO = pytest.mark.xfail(strict=True, reason="보강 목표: 아직 구현 전")


def reason(question, sql):
    return drop_reason({"question": question, "sql": sql})


# ---------------------------------------------------------------------------
# A3. 비교 방향: 질문의 비교 표현과 그 컬럼·값의 SQL 연산자
# ---------------------------------------------------------------------------

A3_WRONG = [
    ("show items with stock above 50", "SELECT * FROM items WHERE stock < 50"),
    ("what is the average price of items with stock above 50?", "SELECT AVG(price) FROM items WHERE stock < 50"),
    ("list orders placed after 2025-06-01", "SELECT * FROM orders WHERE order_date < '2025-06-01'"),
    ("which items cost less than $20?", "SELECT item_name FROM items WHERE price >= 20.00"),
    ("categories whose average price is at least 50, with that average",
     "SELECT category, AVG(price) FROM items GROUP BY category HAVING AVG(price) <= 50"),
    ("show item names priced above the average price",
     "SELECT item_name FROM items WHERE price < (SELECT AVG(price) FROM items)"),
]
A3_RIGHT = [
    ("show items with stock above 50", "SELECT * FROM items WHERE stock > 50"),
    ("which items are not over $20?", "SELECT item_name FROM items WHERE price <= 20.00"),
    ("show items with a stock of 50 or more", "SELECT * FROM items WHERE stock >= 50"),
    ("list orders placed no earlier than 2025-03-01", "SELECT * FROM orders WHERE order_date >= '2025-03-01'"),
    ("list orders from 2025-03-01", "SELECT * FROM orders WHERE order_date >= '2025-03-01'"),
    ("show items with stock between 10 and 20", "SELECT * FROM items WHERE stock BETWEEN 10 AND 20"),
    ("what is the average price of items with stock above 50?", "SELECT AVG(price) FROM items WHERE stock > 50"),
    ("show item names priced above the average price",
     "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items)"),
]


@pytest.mark.parametrize("question, sql", A3_WRONG)
def test_a3_flipped_comparison_is_flagged(question, sql):
    assert reason(question, sql) is not None


@pytest.mark.parametrize("question, sql", A3_RIGHT)
def test_a3_guard_matching_comparison_passes(question, sql):
    assert reason(question, sql) is None


# ---------------------------------------------------------------------------
# A4. 그룹 키: by / per / each 가 가리키는 키 (정렬 "sorted by", 조건 "by customer 1452"는 그룹이 아님)
# ---------------------------------------------------------------------------

# SQL 의 그룹 키가 조건으로 질문에 나와 "키 단어 있음"만으로는 못 잡는 경우
A4_WRONG = [
    ("number of gold membership customers per city",
     "SELECT membership, COUNT(*) FROM customers WHERE membership = 'gold' GROUP BY membership"),
    ("total units ordered per item for customer 1452",
     "SELECT customer_id, SUM(quantity) FROM orders WHERE customer_id = 1452 GROUP BY customer_id"),
    ("how many orders per status have order date 2025-05-01?",
     "SELECT order_date, COUNT(*) FROM orders WHERE order_date = '2025-05-01' GROUP BY order_date"),
]
A4_RIGHT = [
    ("number of customers per city among gold members",
     "SELECT city, COUNT(*) FROM customers WHERE membership = 'gold' GROUP BY city"),
    ("total units ordered per item for customer 1452",
     "SELECT item_id, SUM(quantity) FROM orders WHERE customer_id = 1452 GROUP BY item_id"),
    ("for kitchen and sports items, average price by category",
     "SELECT category, AVG(price) FROM items WHERE category IN ('kitchen', 'sports') GROUP BY category"),
    ("average price by category, sorted by average price descending",
     "SELECT category, AVG(price) FROM items GROUP BY category ORDER BY AVG(price) DESC, category ASC"),
]


@pytest.mark.parametrize("question, sql", A4_WRONG)
def test_a4_group_key_named_in_question_must_match(question, sql):
    assert reason(question, sql) is not None


@pytest.mark.parametrize("question, sql", A4_RIGHT)
def test_a4_guard_matching_group_key_passes(question, sql):
    assert reason(question, sql) is None


# ---------------------------------------------------------------------------
# A5. WHERE / HAVING: 비교가 행 속성에 붙는지 집계 결과에 붙는지
# ---------------------------------------------------------------------------

A5_WRONG = [
    ("average price by category for items priced at least 50",
     "SELECT category, AVG(price) FROM items GROUP BY category HAVING AVG(price) >= 50"),
    ("categories whose average price is at least 50, with that average",
     "SELECT category, AVG(price) FROM items WHERE price >= 50.00 GROUP BY category"),
    ("total stock by category for categories whose total stock exceeds 500",
     "SELECT category, SUM(stock) FROM items WHERE stock > 500 GROUP BY category"),
]
A5_RIGHT = [
    ("average price by category for items priced at least 50",
     "SELECT category, AVG(price) FROM items WHERE price >= 50.00 GROUP BY category"),
    ("categories whose average price is at least 50, with that average",
     "SELECT category, AVG(price) FROM items GROUP BY category HAVING AVG(price) >= 50"),
    ("customers with at least 5 orders and their order counts",
     "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) >= 5"),
    ("count orders with quantity of at least 5 for each customer",
     "SELECT customer_id, COUNT(*) FROM orders WHERE quantity >= 5 GROUP BY customer_id"),
]


@pytest.mark.parametrize("question, sql", A5_WRONG)
def test_a5_where_having_swap_is_flagged(question, sql):
    assert reason(question, sql) is not None


@pytest.mark.parametrize("question, sql", A5_RIGHT)
def test_a5_guard_correct_scope_passes(question, sql):
    assert reason(question, sql) is None


# ---------------------------------------------------------------------------
# A6. 서브쿼리 범위: 조회 대상 조건(바깥)과 기준 집단 조건(안쪽)
# ---------------------------------------------------------------------------

A6_WRONG = [
    ("kitchen items priced above the overall average price",
     "SELECT * FROM items WHERE price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')"),
    ("items priced above the average price of kitchen items",
     "SELECT * FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items)"),
    ("items with stock over 50 priced above the overall average price",
     "SELECT * FROM items WHERE price > (SELECT AVG(price) FROM items WHERE stock > 50)"),
    ("orders with quantity above the average quantity of pending orders",
     "SELECT * FROM orders WHERE status = 'pending' AND quantity > (SELECT AVG(quantity) FROM orders)"),
]
A6_RIGHT = [
    ("kitchen items priced above the overall average price",
     "SELECT * FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items)"),
    ("items priced above the average price of kitchen items",
     "SELECT * FROM items WHERE price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')"),
    ("kitchen items above the average kitchen item price",
     "SELECT * FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')"),
    ("items with stock over 50 priced above the overall average price",
     "SELECT * FROM items WHERE stock > 50 AND price > (SELECT AVG(price) FROM items)"),
]


@pytest.mark.parametrize("question, sql", A6_WRONG)
def test_a6_inner_outer_swap_is_flagged(question, sql):
    assert reason(question, sql) is not None


@pytest.mark.parametrize("question, sql", A6_RIGHT)
def test_a6_guard_correct_scope_passes(question, sql):
    assert reason(question, sql) is None


# ---------------------------------------------------------------------------
# A2. 판정 세 가지(위반 / 검토 필요 / 통과)와 근거·검사하지 못한 항목
# ---------------------------------------------------------------------------

def test_a2_judge_returns_status_evidence_and_unchecked():
    wrong = ca.judge({"question": "what is the average price of kitchen items?",
                      "sql": "SELECT MAX(price) FROM items WHERE category = 'kitchen'"})
    assert wrong.status == "violation" and wrong.reason == "L19" and wrong.evidence
    right = ca.judge({"question": "what is the average price of kitchen items?",
                      "sql": "SELECT AVG(price) FROM items WHERE category = 'kitchen'"})
    assert right.status == "pass" and right.reason is None and right.unchecked == []
    # 비교 표현의 방향을 정할 수 없으면 통과가 아니라 검토 필요다
    vague = ca.judge({"question": "show items with stock in the vicinity of 50",
                      "sql": "SELECT * FROM items WHERE stock > 50"})
    assert vague.status == "review" and vague.unchecked


def test_a2_drop_reason_stays_the_violation_view_of_judge():
    pair = {"question": "what is the average price of kitchen items?",
            "sql": "SELECT AVG(stock) FROM items WHERE category = 'kitchen'"}
    assert drop_reason(pair) == ca.judge(pair).reason == "L20"
    vague = {"question": "show items with stock in the vicinity of 50", "sql": "SELECT * FROM items WHERE stock > 50"}
    assert drop_reason(vague) is None  # 검토 필요는 위반이 아니다
