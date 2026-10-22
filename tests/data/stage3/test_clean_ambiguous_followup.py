"""조합기 학습 문장의 오탐 회귀: 정상 조건과 같은 문형의 잘못된 라벨을 함께 검사한다."""
import pytest

from src.data.stage3.clean_ambiguous import drop_reason, judge

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("question, correct, wrong", [
    ("show item ids with stock <= 81 or more expensive than $10",
     "SELECT item_id FROM items WHERE price > 10.00 OR stock <= 81",
     "SELECT item_id FROM items WHERE price > 10.00 OR stock >= 81"),
    ("show stock for items with stock < 21 and up to $137.99",
     "SELECT stock FROM items WHERE price <= 137.99 AND stock < 21",
     "SELECT stock FROM items WHERE price <= 137.99 AND stock > 21"),
    ("retrieve categories of things with stock < 0 and leaving out kitchen and priced at $166.11 or more",
     "SELECT category FROM items WHERE category != 'kitchen' AND price >= 166.11 AND stock < 0",
     "SELECT category FROM items WHERE category != 'kitchen' AND price < 166.11 AND stock < 0"),
    ("can you pull up the 3 lowest priced items excluding clothing and priced over 60.00?",
     "SELECT * FROM items WHERE category != 'clothing' AND price > 60.00 ORDER BY price ASC, item_id ASC LIMIT 3",
     "SELECT * FROM items WHERE category != 'clothing' AND price < 60.00 ORDER BY price ASC, item_id ASC LIMIT 3"),
    ("show stock for items leaving out office supplies and priced at 170.87 or more",
     "SELECT stock FROM items WHERE category != 'office supplies' AND price >= 170.87",
     "SELECT stock FROM items WHERE category != 'office supplies' AND price < 170.87"),
])
def test_comparison_cue_stays_with_its_condition(question, correct, wrong):
    assert drop_reason({"question": question, "sql": correct}) is None
    assert drop_reason({"question": question, "sql": wrong}) == "L22"


@pytest.mark.parametrize("question, correct, wrong", [
    ("minimum stock by category among items with 16 or fewer in stock, keep groups whose minimum stock is less than 0",
     "SELECT category, MIN(stock) FROM items WHERE stock <= 16 GROUP BY category HAVING MIN(stock) < 0",
     "SELECT category, MIN(stock) FROM items WHERE stock < 0 GROUP BY category HAVING MIN(stock) <= 16"),
    ("maximum price for clothes priced at $5.76 or more, grouped by category",
     "SELECT category, MAX(price) FROM items WHERE price >= 5.76 GROUP BY category",
     "SELECT category, MAX(price) FROM items GROUP BY category HAVING MAX(price) >= 5.76"),
    ("highest stock level for sports gear with > 10 in stock by category",
     "SELECT category, MAX(stock) FROM items WHERE category = 'sports' AND stock > 10 GROUP BY category",
     "SELECT category, MAX(stock) FROM items WHERE category = 'sports' GROUP BY category HAVING MAX(stock) > 10"),
    ("lowest stock level for clothes with no fewer than 85 left by category",
     "SELECT category, MIN(stock) FROM items WHERE stock >= 85 GROUP BY category",
     "SELECT category, MIN(stock) FROM items GROUP BY category HAVING MIN(stock) >= 85"),
    ("show category and mean stock using only the available items",
     "SELECT category, AVG(stock) FROM items WHERE stock > 0 GROUP BY category",
     "SELECT category, AVG(stock) FROM items GROUP BY category HAVING AVG(stock) > 0"),
    ("minimum price by category for things whose names end with mer and below $15",
     "SELECT category, MIN(price) FROM items WHERE item_name LIKE '%mer' AND price < 15.00 GROUP BY category",
     "SELECT category, MIN(price) FROM items WHERE item_name LIKE '%mer' GROUP BY category HAVING MIN(price) < 15"),
])
def test_row_filter_is_not_an_aggregate_threshold(question, correct, wrong):
    assert drop_reason({"question": question, "sql": correct}) is None
    assert drop_reason({"question": question, "sql": wrong}) == "L22"


@pytest.mark.parametrize("question, correct, wrong", [
    ("show stock for products with 0+ in stock and above $42.99",
     "SELECT stock FROM items WHERE price > 42.99 AND stock >= 0",
     "SELECT stock FROM items WHERE price > 42.99 AND stock < 0"),
    ("show stock for items with 0 or fewer in stock",
     "SELECT stock FROM items WHERE stock <= 0",
     "SELECT stock FROM items WHERE stock > 0"),
])
def test_stock_word_does_not_create_an_extra_zero_boundary(question, correct, wrong):
    assert drop_reason({"question": question, "sql": correct}) is None
    assert drop_reason({"question": question, "sql": wrong}) == "L22"


@pytest.mark.parametrize("question, correct, wrong", [
    ("show order ids with quantity below the maximum quantity from orders placed after 2025-03-19, for items 3659 and 3164",
     "SELECT order_id FROM orders WHERE quantity < (SELECT MAX(quantity) FROM orders WHERE item_id IN (3164, 3659) AND order_date > '2025-03-19')",
     "SELECT order_id FROM orders WHERE item_id IN (3164, 3659) AND quantity < (SELECT MAX(quantity) FROM orders WHERE order_date > '2025-03-19')"),
    ("show all details of kitchen items priced above the average price among kitchenware",
     "SELECT * FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')",
     "SELECT * FROM items WHERE category = 'kitchen' AND price < (SELECT AVG(price) FROM items WHERE category = 'kitchen')"),
    ("show quantities above the average quantity of orders waiting to ship",
     "SELECT quantity FROM orders WHERE quantity > (SELECT AVG(quantity) FROM orders WHERE status = 'pending')",
     "SELECT quantity FROM orders WHERE status = 'pending' AND quantity > (SELECT AVG(quantity) FROM orders)"),
    ("show item names for items $40 max with stock above the average stock of sports items",
     "SELECT item_name FROM items WHERE price <= 40.00 AND stock > (SELECT AVG(stock) FROM items WHERE category = 'sports')",
     "SELECT item_name FROM items WHERE price <= 40.00 AND stock < (SELECT AVG(stock) FROM items WHERE category = 'sports')"),
    ("show quantity for orders excluding shipped: quantity equal to the maximum quantity across orders by customer 1870",
     "SELECT quantity FROM orders WHERE status != 'shipped' AND quantity = (SELECT MAX(quantity) FROM orders WHERE customer_id = 1870)",
     "SELECT quantity FROM orders WHERE status != 'shipped' AND quantity < (SELECT MAX(quantity) FROM orders WHERE customer_id = 1870)"),
])
def test_reference_population_and_outer_predicate_do_not_bleed(question, correct, wrong):
    assert drop_reason({"question": question, "sql": correct}) is None
    assert drop_reason({"question": question, "sql": wrong}) is not None


def test_uncertain_population_tail_is_review_but_explicit_wrong_direction_is_flagged():
    question = "show item names above the average price among kitchen items, for items with stock above 50"
    sql = "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items WHERE category = 'kitchen' AND stock > 50)"
    result = judge({"question": question, "sql": sql})
    assert result.status == "review" and result.unchecked
    assert drop_reason({"question": question, "sql": sql.replace("stock > 50", "stock < 50")}) == "L22"


def test_like_fragment_does_not_mask_the_negative_comparison_word():
    question = 'show item names for things with names containing gpi, no cheaper than $15 where stock is below the highest stock of clothes where name like no*'
    sql = "SELECT item_name FROM items WHERE item_name LIKE '%gpi%' AND price >= 15.00 AND stock < (SELECT MAX(stock) FROM items WHERE item_name LIKE 'no%' AND category = 'clothing')"
    assert drop_reason({"question": question, "sql": sql}) is None
    assert drop_reason({"question": question, "sql": sql.replace("price >= 15.00", "price < 15.00")}) == "L22"


def test_numeric_minimum_is_not_a_second_scalar_reference():
    # 학습 틀의 reference value 도입부와 숫자 min 경계가 함께 있는 경우다.
    question = "show quantities below a reference value defined as the mean quantity from orders with quantity of min 8 units"
    sql = "SELECT quantity FROM orders WHERE quantity < (SELECT AVG(quantity) FROM orders WHERE quantity >= 8)"
    assert drop_reason({"question": question, "sql": sql}) is None
    assert drop_reason({"question": question, "sql": sql.replace("quantity >= 8", "quantity < 8")}) == "L22"


def test_repeated_number_keeps_inner_and_outer_directions_separate():
    question = "show order ids for orders with quantity above 5 and quantity at least the average quantity of orders with quantity under 5"
    correct = "SELECT order_id FROM orders WHERE quantity > 5 AND quantity >= (SELECT AVG(quantity) FROM orders WHERE quantity < 5)"
    wrong = "SELECT order_id FROM orders WHERE quantity < 5 AND quantity >= (SELECT AVG(quantity) FROM orders WHERE quantity > 5)"
    assert drop_reason({"question": question, "sql": correct}) is None
    assert drop_reason({"question": question, "sql": wrong}) == "L22"
