"""Finite template contracts, not a semantic LLM call per generated question."""
import copy
import json
import random
import sqlite3
from pathlib import Path
from string import Formatter

import pytest

from src.data.stage3 import sql_ast as a, sql_ast, sql_rules
from src.data.stage3.compose import _compose_aggregate as compose
from src.data.stage3.compose import Composer, question_order_sql, literal_problems, value_keys

pytestmark = pytest.mark.unit
TEMPLATES = json.loads(Path("data/stage3/composer/advanced.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("table, column", [("items", "price"), ("items", "stock"),
                                          ("orders", "quantity"), ("orders", "order_date")])
@pytest.mark.parametrize("direction", ["ASC", "DESC"])
def test_every_new_order_template_uses_primary_key_ascending(table, column, direction):
    query = sql_ast.Query(table, "*", order=(column, direction), limit=3)
    primary = {"items": "item_id", "orders": "order_id"}[table]
    expected = f"ORDER BY {column} {direction}, {primary} ASC LIMIT 3"
    assert query.render().endswith(expected)
    assert sql_ast.parse(query.render()).render() == query.render()
    assert sql_ast.from_dict(sql_ast.to_dict(query)).render() == query.render()
    assert not sql_ast.violations(query)


@pytest.mark.parametrize("sql", [
    "SELECT stock FROM items ORDER BY stock DESC LIMIT 1",
    "SELECT * FROM orders ORDER BY order_date DESC, order_id DESC LIMIT 3",
    "SELECT * FROM orders ORDER BY order_date ASC, order_id ASC LIMIT 3",
])
def test_old_sql_is_read_without_silently_changing_its_meaning(sql):
    query = sql_ast.parse(sql)
    assert query.render() == sql
    assert sql_ast.from_dict(sql_ast.to_dict(query)).render() == sql


def test_in_value_reordering_does_not_change_an_explicit_tie_request():
    sql = ("SELECT order_id FROM orders WHERE quantity IN (1, 6) "
           "ORDER BY order_date DESC, order_id DESC LIMIT 3")
    record = sql_ast.to_dict(sql_ast.parse(sql))
    question = "order ids for quantity 6 or 1, newest dates first and highest ids first on ties, first 3"
    result = question_order_sql(record, question)
    assert "quantity IN (6, 1)" in result
    assert result.endswith("ORDER BY order_date DESC, order_id DESC LIMIT 3")


def test_scoring_does_not_ignore_a_wrong_tie_direction():
    gold = "SELECT * FROM orders ORDER BY order_date DESC, order_id ASC LIMIT 3"
    pred = gold.replace("order_id ASC", "order_id DESC")
    assert not sql_rules.same_sql(gold, pred)
    assert sql_rules.error_parts(gold, pred) == ["order"]


def test_limit_one_and_all_extremal_rows_have_different_contracts():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE items(item_id INTEGER, item_name TEXT, price REAL)")
    db.executemany("INSERT INTO items VALUES (?, ?, ?)", [(8, "a", 20), (2, "b", 20), (3, "c", 10)])
    first = sql_ast.Query("items", "item_name", order=("price", "DESC"), limit=1).render()
    tied = "SELECT item_name FROM items WHERE price = (SELECT MAX(price) FROM items)"
    assert db.execute(first).fetchall() == [("b",)]
    assert set(db.execute(tied)) == {("a",), ("b",)}
    db.close()


PLACEHOLDERS = {
    "aggregate": {"aggregate", "rows"}, "group": {"aggregate", "rows", "group"},
    "having": {"aggregate", "comparison", "value"},
    "subquery": {"projection", "column", "comparison", "aggregate", "reference"},
    "subquery_default": {"projection", "column", "comparison", "aggregate"},
    "group_default": {"subject", "rows", "group"},
}
FRAME_CASES = [(role, split, i, text) for role in PLACEHOLDERS
               for split, texts in TEMPLATES[role].items() for i, text in enumerate(texts)]


@pytest.mark.parametrize("role, split, index, text", FRAME_CASES)
def test_each_frame_has_all_semantically_required_slots(role, split, index, text):
    fields = {field for _, field, _, _ in Formatter().parse(text) if field}
    assert fields == PLACEHOLDERS[role]
    # These typed slots can be singular attributes or plural rows/count subjects.
    # Their enclosing frame must not assume a fixed grammatical number.
    assert "what are {projection}" not in text.lower()
    assert "what is the {subject}" not in text.lower()
    assert "a {subject} result" not in text.lower()


@pytest.mark.parametrize("split", ["train", "eval"])
def test_filtered_reference_population_never_uses_implicit_global_template(split):
    # The category applies only to the reference population, the stock test only to returned rows.
    sql = ("SELECT item_name FROM items WHERE stock > 17 AND price > "
           "(SELECT AVG(price) FROM items WHERE category = 'kitchen')")
    record = a.to_dict(a.parse(sql))
    for frame in TEMPLATES["subquery"][split]:
        templates = copy.deepcopy(TEMPLATES)
        templates["subquery"][split] = [frame]
        templates["subquery_default"][split] = ["INVALID GLOBAL REFERENCE"]
        for seed in range(10):
            question, _ = compose(record, random.Random(seed), Composer.load(split), split, templates)
            assert "invalid" not in question
            assert "greater than" in question
            # Reference-first requests are valid too. The OUTER stock filter
            # must remain inside the projection phrase, and kitchen inside
            # the independent reference phrase, in the frame's chosen order.
            if frame.index("{projection}") < frame.index("{reference}"):
                assert question.index("17") < question.index("kitchen")
            else:
                assert question.index("kitchen") < question.index("17")
            assert "price" in question and ("average" in question or "mean" in question)


@pytest.mark.parametrize("function, cue", [("SUM", "sum|total|combined"), ("AVG", "average|mean"),
                                          ("MIN", "minimum|lowest|smallest"), ("MAX", "maximum|highest|largest")])
def test_every_aggregate_expression_has_the_right_operation_and_column(function, cue):
    import re
    for split in ("train", "eval"):
        for expression in TEMPLATES["aggregates"][function][split]:
            assert "{column}" in expression
            assert re.search(rf"\b(?:{cue})\b", expression)


def test_where_having_and_group_output_survive_each_group_and_having_frame():
    sql = ("SELECT category, AVG(price) FROM items WHERE stock >= 17 "
           "GROUP BY category HAVING AVG(price) < 42")
    record = a.to_dict(a.parse(sql))
    for split in ("train", "eval"):
        for group in TEMPLATES["group"][split]:
            for having in TEMPLATES["having"][split]:
                templates = copy.deepcopy(TEMPLATES)
                templates.pop("group_default", None)
                templates["group"][split] = [group]
                templates["having"][split] = [having]
                question, _ = compose(record, random.Random(2), Composer.load(split), split, templates)
                assert "category" in question and "17" in question
                assert "price" in question and "less than 42" in question
                assert question.index("17") < question.index("less than 42")


def test_global_average_can_be_implicit_but_outer_filter_is_preserved():
    sql = "SELECT item_name FROM items WHERE category = 'kitchen' AND price > (SELECT AVG(price) FROM items)"
    record = a.to_dict(a.parse(sql))
    questions = [compose(record, random.Random(seed))[0] for seed in range(40)]
    assert all("kitchen" in q and "price" in q and "greater than" in q for q in questions)
    assert any(q.rstrip("?").endswith(("average price", "mean price")) for q in questions)
    assert any(" for " in q[q.index("greater than"):] for q in questions)


def test_low_stock_is_a_fixed_threshold_not_an_arbitrary_number():
    composer = Composer.load()
    records = [sql_ast.to_dict(sql_ast.Query("items", "*", (sql_ast.Cond("stock", "<=", n),)))
               for n in (10, 11)]
    questions = [[composer.compose(r, random.Random(seed))[0] for seed in range(100)] for r in records]
    assert any("low stock" in q or "running low on stock" in q for q in questions[0])
    assert not any("low stock" in q or "running low on stock" in q for q in questions[1])
    lexicon = json.loads(Path("data/stage3/composer/lexicon.json").read_text(encoding="utf-8"))
    assert all(not literal_problems(q, records[0], value_keys(lexicon)) for q in questions[0])


@pytest.mark.parametrize("aggregate, subject", [("AVG(price)", "price"), ("SUM(stock)", "stock"), ("COUNT(*)", "items")])
def test_implicit_group_templates_use_the_policy_default_operation(aggregate, subject):
    sql = f"SELECT category, {aggregate} FROM items WHERE stock >= 17 GROUP BY category"
    record = a.to_dict(a.parse(sql))
    templates = copy.deepcopy(TEMPLATES)
    # A sentinel separates the default wording path from explicitly named aggregates.
    templates["group_default"]["train"] = ["implicit: {subject} by {group} for {rows}"]
    questions = [compose(record, random.Random(seed), templates=templates)[0] for seed in range(60)]
    implicit = [q for q in questions if q.startswith("implicit:")]
    assert implicit
    assert all(q.startswith(f"implicit: {subject} by category") and "17" in q for q in implicit)
