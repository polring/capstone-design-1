"""3단계 라벨 판정: 고급(집계·그룹·서브쿼리) 쌍의 L19~L24 와 일반 조회의 L19 수량 합계."""
import pytest

from src.data.stage3.clean_ambiguous import REASONS, drop_reason
from src.data.stage3 import clean_ambiguous as ca

pytestmark = pytest.mark.unit


def reason(question, sql):
    return drop_reason({"question": question, "sql": sql})


@pytest.mark.parametrize("question, sql", [
    ("what is the average price of kitchen items?", "SELECT AVG(price) FROM items WHERE category = 'kitchen'"),
    ("how many orders did customer 1452 place?", "SELECT COUNT(*) FROM orders WHERE customer_id = 1452"),
    ("how many units did customer 1452 order?", "SELECT SUM(quantity) FROM orders WHERE customer_id = 1452"),
    ("what do item prices total where stock equals 39?", "SELECT SUM(price) FROM items WHERE stock = 39"),
    ("show each category and its price, grouped by category",
     "SELECT category, AVG(price) FROM items GROUP BY category"),
    ("display each membership tier and its customers", "SELECT membership, COUNT(*) FROM customers GROUP BY membership"),
    ("list non-sports categories and their total stock where the total exceeds 3036.",
     "SELECT category, SUM(stock) FROM items WHERE category != 'sports' GROUP BY category HAVING SUM(stock) > 3036"),
    ("list categories and their lowest prices for stock between 10 and 25 inclusive, cheapest minimum first.",
     "SELECT category, MIN(price) FROM items WHERE stock BETWEEN 10 AND 25 GROUP BY category "
     "ORDER BY MIN(price) ASC, category ASC"),
    ("what are the units ordered per order date, sorted by total units descending and date ascending for ties?",
     "SELECT order_date, SUM(quantity) FROM orders GROUP BY order_date ORDER BY SUM(quantity) DESC, order_date ASC"),
    ("which 3 categories have the lowest average price?",
     "SELECT category, AVG(price) FROM items GROUP BY category ORDER BY AVG(price) ASC, category ASC LIMIT 3"),
    ("which category leads in item count?",
     "SELECT category, COUNT(*) FROM items GROUP BY category ORDER BY COUNT(*) DESC, category ASC LIMIT 1"),
    ("kitchen items more expensive than the average kitchen item price",
     "SELECT * FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')"),
    ("which items cost more than the average price of items with stock over 50?",
     "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items WHERE stock > 50)"),
    ("show all details of items at the highest price of the things with low stock",
     "SELECT * FROM items WHERE price = (SELECT MAX(price) FROM items WHERE stock <= 10)"),
    ("what is the total price of the items named steel partition or outdoor racket?",
     "SELECT SUM(price) FROM items WHERE item_name IN ('outdoor racket', 'steel partition')"),
])
def test_consistent_advanced_labels_pass(question, sql):
    assert reason(question, sql) is None


@pytest.mark.parametrize("question, sql, expected", [
    # L19: 함수
    ("what is the average price of kitchen items?", "SELECT MAX(price) FROM items WHERE category = 'kitchen'", "L19"),
    ("how many units did customer 1452 order?", "SELECT COUNT(*) FROM orders WHERE customer_id = 1452", "L19"),
    ("how many customers are in chicago?", "SELECT COUNT(*) FROM orders WHERE customer_id = 1452", "L19"),
    # L20: 컬럼
    ("what is the average price of kitchen items?", "SELECT AVG(stock) FROM items WHERE category = 'kitchen'", "L20"),
    # L21: 그룹
    ("what is the average price per category?", "SELECT AVG(price) FROM items", "L21"),
    ("what is the average price per category?", "SELECT city, COUNT(*) FROM customers GROUP BY city", "L19"),
    ("show each category and its average price", "SELECT status, AVG(quantity) FROM orders GROUP BY status", "L20"),
    # L22: SQL 에 없는 조건 값
    ("what is the average price of kitchen items?", "SELECT AVG(price) FROM items", "L22"),
    # L23: 기준 집단
    ("which items cost more than the average price of items with stock over 50?",
     "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items)", "L23"),
    ("which kitchen items cost more than their average price?",
     "SELECT item_name FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items)", "L23"),
    # L24: 정렬·LIMIT
    ("which 3 categories have the lowest average price?",
     "SELECT category, AVG(price) FROM items GROUP BY category ORDER BY AVG(price) DESC, category ASC LIMIT 3", "L24"),
    ("which 3 categories have the lowest average price?",
     "SELECT category, AVG(price) FROM items GROUP BY category ORDER BY AVG(price) ASC, category ASC LIMIT 5", "L24"),
    ("list categories with their average price, highest first",
     "SELECT category, AVG(price) FROM items GROUP BY category", "L24"),
    ("list categories with their average price",
     "SELECT category, AVG(price) FROM items GROUP BY category ORDER BY AVG(price) ASC, category ASC", "L24"),
])
def test_inconsistent_advanced_labels_are_flagged(question, sql, expected):
    assert reason(question, sql) == expected
    assert expected in REASONS


@pytest.mark.parametrize("question, sql, expected", [
    # 그 테이블에서 그룹 키가 될 수 없는 "per X"는 단가·단위 표현이다
    ("what is the average price per item in the kitchen category?",
     "SELECT AVG(price) FROM items WHERE category = 'kitchen'", None),
    ("what is the average number of units per order?", "SELECT AVG(quantity) FROM orders", None),
    ("what is the average cost per product among sports items?",
     "SELECT AVG(price) FROM items WHERE category = 'sports'", None),
    ("what is the average stock sorted by price?", "SELECT AVG(stock) FROM items", None),
    ("how many units were ordered by customer 1452?", "SELECT SUM(quantity) FROM orders WHERE customer_id = 1452", None),
    # 그룹 키가 될 수 있으면 그룹 요청이다
    ("how many units were ordered per item?", "SELECT SUM(quantity) FROM orders", "L21"),
    ("what is the total stock for each category?", "SELECT SUM(stock) FROM items", "L21"),
    ("how many customers are there per city?", "SELECT COUNT(*) FROM customers", "L21"),
    ("how many units were ordered per item?", "SELECT item_id, SUM(quantity) FROM orders GROUP BY item_id", None),
])
def test_per_phrase_is_group_request_only_for_group_keys(question, sql, expected):
    assert reason(question, sql) == expected


def test_subquery_output_reuses_row_rules_without_existence_check():
    sql = "SELECT item_id FROM items WHERE price > (SELECT AVG(price) FROM items)"
    assert reason("which items cost more than the average price?", sql) == "L3"
    assert reason("which item ids cost more than the average price?", sql) is None
    same_column = "SELECT price FROM items WHERE price > (SELECT AVG(price) FROM items)"
    assert reason("show the prices above the average price", same_column) is None


@pytest.mark.parametrize("question, sql, expected", [
    ("how much stock is there for the kitchen items?", "SELECT stock FROM items WHERE category = 'kitchen'", "L19"),
    ("the products under $20.00, how many units are left?", "SELECT stock FROM items WHERE price < 20.00", "L19"),
    ("how much stock is there for the item named steel partition?",
     "SELECT stock FROM items WHERE item_name = 'steel partition'", None),
    ("how many units are left for each kitchen item?", "SELECT stock FROM items WHERE category = 'kitchen'", None),
    ("how much stock is there for the most expensive kitchen item?",
     "SELECT stock FROM items WHERE category = 'kitchen' ORDER BY price DESC, item_id ASC LIMIT 1", None),
])
def test_plain_units_question_over_many_rows_needs_sum(question, sql, expected):
    assert reason(question, sql) == expected


@pytest.mark.parametrize("question, sql", [
    ("orders of 6 units or fewer", "SELECT * FROM orders WHERE quantity <= 6"),
    ("show items at or below 20 dollars", "SELECT * FROM items WHERE price <= 20.00"),
    ("show items above 20 and above the average price", "SELECT * FROM items WHERE price > 20.00 AND price > (SELECT AVG(price) FROM items)"),
    ("list orders before '2025-06-01'", "SELECT * FROM orders WHERE order_date < '2025-06-01'"),
    ("show items excluding kitchen or priced under 20", "SELECT * FROM items WHERE category != 'kitchen' OR price < 20.00"),
])
def test_a3_comparison_boundaries_do_not_borrow_other_clause(question, sql):
    assert reason(question, sql) is None


@pytest.mark.parametrize("question, sql", [
    ("orders of 6 units or fewer", "SELECT * FROM orders WHERE quantity >= 6"),
    ("show items at or below 20 dollars", "SELECT * FROM items WHERE price >= 20.00"),
    ("list orders before '2025-06-01'", "SELECT * FROM orders WHERE order_date > '2025-06-01'"),
])
def test_a3_natural_comparisons_reject_opposite_direction(question, sql):
    assert reason(question, sql) == "L22"


@pytest.mark.parametrize("question, correct, wrong", [
    ("average price by category for items under 75, keeping averages at least 40",
     "SELECT category, AVG(price) FROM items WHERE price < 75.00 GROUP BY category HAVING AVG(price) >= 40",
     "SELECT category, AVG(price) FROM items WHERE price >= 40.00 GROUP BY category HAVING AVG(price) < 75"),
    ("sum of prices by category after filtering to price below 80",
     "SELECT category, SUM(price) FROM items WHERE price < 80.00 GROUP BY category",
     "SELECT category, SUM(price) FROM items GROUP BY category HAVING SUM(price) < 80"),
    ("total stock by category, excluding groups whose total stock exceeds 500",
     "SELECT category, SUM(stock) FROM items GROUP BY category HAVING SUM(stock) <= 500",
     "SELECT category, SUM(stock) FROM items GROUP BY category HAVING SUM(stock) > 500"),
])
def test_a5_row_filters_and_aggregate_references(question, correct, wrong):
    assert reason(question, correct) is None
    assert reason(question, wrong) == "L22"


@pytest.mark.parametrize("question, correct, wrong", [
    ("use orders with quantity above 2 to find the maximum quantity, then return order ids for orders with quantity below 8 and above that value",
     "SELECT order_id FROM orders WHERE quantity < 8 AND quantity > (SELECT MAX(quantity) FROM orders WHERE quantity > 2)",
     "SELECT order_id FROM orders WHERE quantity > 2 AND quantity > (SELECT MAX(quantity) FROM orders WHERE quantity < 8)"),
    ("list item names for kitchen items above their average price",
     "SELECT item_name FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')",
     "SELECT item_name FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items)"),
    ("list item names for kitchen items priced above the average kitchen item price",
     "SELECT item_name FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')",
     "SELECT item_name FROM items WHERE category = 'sports' AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')"),
])
def test_a6_reference_first_and_repeated_population(question, correct, wrong):
    assert reason(question, correct) is None
    assert reason(question, wrong) is not None


def test_a6_unknown_reference_direction_is_reviewed():
    pair = {"question": "show item names priced near the average price",
            "sql": "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items)"}
    result = ca.judge(pair)
    assert result.status == "review" and result.unchecked
    assert ca.drop_reason(pair) is None


def test_a5_same_number_keeps_row_and_aggregate_operators_separate():
    question = "sum quantity by order date for orders with quantity less than 9; return dates and sums greater than 9"
    correct = "SELECT order_date, SUM(quantity) FROM orders WHERE quantity < 9 GROUP BY order_date HAVING SUM(quantity) > 9"
    assert reason(question, correct) is None
    assert reason(question, correct.replace("SUM(quantity) > 9", "SUM(quantity) < 9")) == "L22"


def test_a3_nested_negation_is_reviewed_without_borrowing_row_boundary():
    pair = {"question": "total stock by category for items with stock at least 25; omit categories whose total isn't above 25",
            "sql": "SELECT category, SUM(stock) FROM items WHERE stock >= 25 GROUP BY category HAVING SUM(stock) > 25"}
    assert ca.judge(pair).status == "review"
    assert ca.drop_reason(pair) is None


@pytest.mark.parametrize("question, sql", [
    ("list each order date and its smallest order quantity greater than 6",
     "SELECT order_date, MIN(quantity) FROM orders WHERE quantity > 6 GROUP BY order_date"),
    ("total item prices below 100 separately for each category",
     "SELECT category, SUM(price) FROM items WHERE price < 100.00 GROUP BY category"),
])
def test_a5_unattached_attribute_modifier_requires_review(question, sql):
    result = ca.judge({"question": question, "sql": sql})
    assert result.status == "review" and result.unchecked


def test_a5_daily_total_without_repeating_aggregate_word():
    question = "give me each order date through 2025-01-26 where more than 45 units were ordered, along with the total units"
    sql = "SELECT order_date, SUM(quantity) FROM orders WHERE order_date <= '2025-01-26' GROUP BY order_date HAVING SUM(quantity) > 45"
    assert reason(question, sql) is None
    assert reason(question, sql.replace("SUM(quantity) > 45", "SUM(quantity) < 45")) == "L22"


@pytest.mark.parametrize("question, sql", [
    ("how many items cost at most 20 per item?", "SELECT COUNT(*) FROM items WHERE price <= 20.00"),
    ("average price by category, sorted by average price descending",
     "SELECT category, AVG(price) FROM items GROUP BY category ORDER BY AVG(price) DESC, category ASC"),
    ("how many orders were placed by customer 1452?", "SELECT COUNT(*) FROM orders WHERE customer_id = 1452"),
])
def test_a4_units_sort_and_customer_filter_are_not_group_keys(question, sql):
    assert reason(question, sql) is None
