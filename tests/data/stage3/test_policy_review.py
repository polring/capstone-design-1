"""Regressions found while reviewing the stage-3 GPT pilot."""
import pytest

from src.data.stage3.clean_ambiguous import drop_reason

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("question, sql", [
    ("list item_ids for items priced at most 173.78", "SELECT item_id FROM items WHERE price <= 173.78"),
    ("return item_names for clothing items priced at most 120.00",
     "SELECT item_name FROM items WHERE category = 'clothing' AND price <= 120.00"),
])
def test_column_spellings_are_output_cues(question, sql):
    assert drop_reason({"question": question, "sql": sql}) is None


@pytest.mark.parametrize("question, expected", [
    ("i am checking orders dated 2025-06-22; please return their order dates", "L1"),
    ("i am checking the exact customer name percy miller; which names match it?", "L1"),
    ("i am checking customers named percy miller; list their matching names", "L1"),
    ("i am checking customer records; does a customer named percy miller exist?", None),
])
def test_background_checking_is_not_existence_intent(question, expected):
    sql = ("SELECT order_date FROM orders WHERE order_date = '2025-06-22'" if "order dates" in question
           else "SELECT name FROM customers WHERE name = 'percy miller'")
    assert drop_reason({"question": question, "sql": sql}) == expected


@pytest.mark.parametrize("question,sql",[
    ("Does any item have a price of 36.86?", "SELECT price FROM items WHERE price = 36.86"),
    ("Does item_id 3041 appear in any orders?", "SELECT item_id FROM orders WHERE item_id = 3041"),
    ("Is 'jonathan price' present in the customer records?", "SELECT name FROM customers WHERE name = 'jonathan price'"),
])
def test_boolean_existence_phrasings_are_not_value_return_requests(question,sql):
    assert drop_reason({"question":question,"sql":sql}) is None
