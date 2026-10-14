import pytest

from src.data.stage2 import sql_rules

pytestmark = pytest.mark.unit

IN_SQL = "SELECT * FROM items WHERE category IN ('kitchen', 'sports') AND price < 20.00"


def test_same_sql_ignores_in_value_and_condition_order():
    """EM: IN 값 순서(L8: 질문 순서)와 AND·OR 조건 순서는 무시한다"""
    assert sql_rules.same_sql(IN_SQL, "SELECT * FROM items WHERE category IN ('sports', 'kitchen') AND price < 20.00")
    assert sql_rules.same_sql(IN_SQL, "SELECT * FROM items WHERE price < 20.00 AND category IN ('kitchen', 'sports')")
    assert not sql_rules.same_sql(IN_SQL, "SELECT * FROM items WHERE category IN ('kitchen', 'sports') OR price < 20.00")
    assert not sql_rules.same_sql(IN_SQL, None)


def test_same_sql_keeps_literal_format_strict():
    """정규형과 표기가 다른 리터럴(price 소수 둘째 자리)은 같은 값이어도 오답이다"""
    assert not sql_rules.same_sql(IN_SQL, "SELECT * FROM items WHERE category IN ('kitchen', 'sports') AND price < 20")


def test_parse_sql_and_malformed():
    g = sql_rules.parse_sql("SELECT name FROM customers WHERE membership = 'gold' AND city = 'boston'")
    assert g["table"] == "customers" and g["select"] == "name" and g["where_col"] == "city+membership"
    assert sql_rules.parse_sql("SELECT * FROM items")["where_col"] == "-"
    assert sql_rules.parse_sql("SELECT * FROM items WHERE category = 'a' AND price < 1 OR stock = 2") is None
    assert sql_rules.error_parts(IN_SQL, "SELECT FROM") == ["malformed"]
    # 그 테이블에 없는 컬럼을 쓴 예측은 오류 없이 오답(malformed)이다
    bad = "SELECT * FROM items WHERE membership = 'gold'"
    assert sql_rules.parse_sql(bad) is None and not sql_rules.same_sql(IN_SQL, bad)
    assert sql_rules.error_parts(IN_SQL, bad) == ["malformed"]
    assert not sql_rules.multi_match("kitchen items", IN_SQL, bad)


def test_error_parts_names_what_differs():
    gold = "SELECT item_name FROM items WHERE item_name LIKE '%isc%' OR price BETWEEN 14.99 AND 25.00 OR stock = 98"
    pred = "SELECT category FROM items WHERE price BETWEEN 14.99 AND 25.00 OR stock = 98"
    assert sql_rules.error_parts(gold, pred) == ["select", "where_col"]
    assert sql_rules.error_parts("SELECT * FROM items WHERE stock >= 20", "SELECT * FROM items WHERE stock = 20") == ["op"]
    assert sql_rules.error_parts("SELECT * FROM items WHERE stock = 20 ORDER BY price ASC LIMIT 3",
                                 "SELECT * FROM items WHERE stock = 21 ORDER BY price DESC") == ["literal", "order", "limit"]


def test_multi_match_swaps_star_and_ident_only_without_cue():
    """SELECT 단서가 없으면 * ↔ 식별 컬럼을 정답으로 인정한다. 존재 확인과 단서 있는 질문은 아니다"""
    gold = "SELECT * FROM items WHERE category = 'kitchen' AND price < 20.00"
    ident = "SELECT item_name FROM items WHERE category = 'kitchen' AND price < 20.00"
    assert sql_rules.select_ambiguous("kitchen stuff under $20", gold)
    assert sql_rules.multi_match("kitchen stuff under $20", gold, ident)
    assert not sql_rules.multi_match("all details of kitchen items under $20", gold, ident)
    assert not sql_rules.select_ambiguous("is there an item called kayak?",
                                          "SELECT item_name FROM items WHERE item_name = 'kayak'")


def test_label_mismatch_uses_stage2_label_rules():
    assert sql_rules.label_mismatch("what is the price of the item priced 9.66?",
                                    "SELECT price FROM items WHERE price = 9.66")  # L1
    assert not sql_rules.label_mismatch("is there an item priced 9.66?", "SELECT price FROM items WHERE price = 9.66")


def test_breakdowns_from_pair_meta_and_sql():
    pair = {"question": "q", "sql": "SELECT * FROM items WHERE category IN ('a', 'b') AND stock > 0 ORDER BY price ASC "
                                     "LIMIT 3", "persona": "keyword", "eval_slot": "cond:items.stock > 0"}
    b = sql_rules.breakdowns(pair, {"struct_combo": 1})
    assert b == {"persona": "keyword", "struct_combo": "LIKE × ORDER BY", "eval_slot": "cond", "table": "items",
                 "conds": "2개 AND +IN", "order": "ORDER BY + LIMIT n"}
    assert list(b) == [k for k in sql_rules.BREAKDOWNS if k in b]
