import pytest

from src.data.stage3.sql_ast import Cond, Query, canonical, features, parse, violations

pytestmark = pytest.mark.unit


def Q(table, select, conds=(), conn=None, order=None, limit=None):
    return canonical(Query(table, select, tuple(conds), conn, order, limit))


def test_render_canonical_order_and_formats():
    """AND 조건은 스키마 순서, price 는 소수 둘째 자리, ASC 명시"""
    q = Q("items", "item_name",
          [Cond("price", "<", 50.0), Cond("category", "=", "kitchen")], "AND", ("price", "ASC"), 3)
    assert q.render() == ("SELECT item_name FROM items WHERE category = 'kitchen' AND price < 50.00 "
                          "ORDER BY price ASC, item_id ASC LIMIT 3")
    assert violations(q) == []


def test_in_values_sorted_and_or_in_schema_order():
    """정규형: IN 값은 값 순서, OR 조건은 AND 처럼 스키마 컬럼 순서"""
    q = Q("customers", "*", [Cond("city", "IN", ("denver", "boston"))])
    assert q.render() == "SELECT * FROM customers WHERE city IN ('boston', 'denver')"
    assert violations(q) == []
    q = Q("items", "*", [Cond("stock", ">", 5), Cond("category", "IN", ("sports", "kitchen"))], "OR")
    assert q.render() == "SELECT * FROM items WHERE category IN ('kitchen', 'sports') OR stock > 5"
    assert violations(q) == []
    q = Q("orders", "*", [Cond("status", "=", "pending"), Cond("customer_id", "IN", (1203, 1102))], "AND")
    assert q.render() == "SELECT * FROM orders WHERE customer_id IN (1102, 1203) AND status = 'pending'"
    assert violations(q) == []


@pytest.mark.parametrize("sql", [
    "SELECT * FROM items WHERE price BETWEEN 10.00 AND 50.00 AND stock > 5",
    "SELECT name FROM customers WHERE name LIKE 'ma%' AND city != 'chicago'",
    "SELECT stock FROM items ORDER BY stock DESC LIMIT 1",
    "SELECT order_id FROM orders WHERE status IN ('pending', 'shipped')",
    "SELECT * FROM items WHERE category IN ('kitchen', 'office supplies') OR price < 10.00",
    "SELECT * FROM orders WHERE customer_id IN (1102, 1203) AND status = 'pending'",
    "SELECT customer_id FROM customers WHERE name IN ('amy', 'jane', 'zoe')",
    "SELECT quantity FROM orders WHERE customer_id = 1430 ORDER BY quantity ASC",
])
def test_parse_roundtrip(sql):
    """정규형 SQL 은 parse → render 로 그대로 돌아온다"""
    assert parse(sql).render() == sql


def test_parse_rejects_mixed_and_or():
    """AND 와 OR 를 섞은 SQL 은 문법 밖"""
    with pytest.raises(ValueError):
        parse("SELECT * FROM customers WHERE city = 'boston' OR city = 'denver' AND membership = 'gold'")


@pytest.mark.parametrize("q, msg", [
    (Query("customers", "*", (Cond("city", "<", "boston"),)), "불가"),
    (Query("items", "*", (Cond("price", ">", 30.0), Cond("price", "<", 50.0)), "AND"), "BETWEEN"),
    (Query("items", "*", (Cond("price", "BETWEEN", 50.0, 10.0),)), "하한"),
    (Query("customers", "*", (Cond("city", "=", "boston"), Cond("city", "=", "denver")), "OR"), "IN"),
    (Query("items", "*", (Cond("price", "IN", (1.5, 2.5)),)), "불가"),
    (Query("items", "*", (Cond("category", "IN", ("kitchen",)),)), "IN 값"),
    (Query("customers", "city", (Cond("city", "IN", ("boston", "denver")),)), "IN 일 때 SELECT"),
    (Query("items", "*", (Cond("category", "=", "kitchen"),), limit=3), "LIMIT"),
    (Query("items", "*", (Cond("stock", "=", 3),), order=("stock", "ASC")), "고정"),
    (Query("customers", "*", order=("city", "ASC")), "정렬 불가"),
    (Query("items", "category", (Cond("category", "=", "kitchen"), Cond("stock", ">", 3)), "AND"), "SELECT"),
    (Query("items", "category", (Cond("category", "=", "kitchen"),), order=("stock", "ASC")), "존재 확인"),
    (Query("customers", "*", (Cond("name", "LIKE", "ma"),)), "LIKE"),
    (Query("items", "*", (Cond("stock", ">", 3), Cond("category", "=", "kitchen")), "AND"), "정규형"),
])
def test_violations(q, msg):
    """구조 규칙 위반을 잡는다"""
    v = violations(q)
    assert any(msg in x for x in v), v


def test_single_eq_select_same_col_allowed():
    """조건 1개 = 에서 SELECT 가 조건 컬럼이면 1단계 존재 확인(L1)으로 허용"""
    assert violations(Q("items", "price", [Cond("price", "=", 12.5)])) == []


@pytest.mark.parametrize("sql", [
    "SELECT * FROM items",
    "SELECT * FROM items WHERE item_name = 'wireless mouse' AND stock > 3",
    "SELECT * FROM customers WHERE name IN ('amy', 'jane') AND city = 'boston'",
    "SELECT * FROM customers WHERE city = 'boston' OR membership != 'gold'",
    "SELECT * FROM orders WHERE customer_id = 1102 OR quantity > 5",
    "SELECT * FROM items WHERE category = 'kitchen' OR price < 9.00 ORDER BY price ASC, item_id ASC LIMIT 3",
])
def test_valid_queries_are_not_rejected_for_sampling_preferences(sql):
    query = parse(sql)
    assert violations(canonical(query)) == []


def test_features():
    """구조 홀드아웃 판정용 특징"""
    q = Q("orders", "*", [Cond("quantity", "BETWEEN", 2, 5)], order=("quantity", "DESC"), limit=1)
    f = features(q)
    assert {("table", "orders"), ("op", "BETWEEN"), ("order", True), ("limit", True), ("n", 1)} <= f
@pytest.mark.unit
def test_date_range_and_stable_ordering():
    from src.data.stage3.sql_ast import Cond, Query, parse, violations

    query = Query("orders", "*", (Cond("order_date", "BETWEEN", "2025-12-01", "2026-01-31"),),
                  order=("order_date", "DESC"), limit=5)
    assert query.render().endswith("ORDER BY order_date DESC, order_id ASC LIMIT 5")
    assert parse(query.render()).render() == query.render()
    assert violations(query) == []
    assert violations(Query("orders", "*", (Cond("order_date", "=", "2025-02-30"),)))

