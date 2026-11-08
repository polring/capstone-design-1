import json
import random
import sqlite3
from pathlib import Path

import pytest

from src.data.stage3 import sql_ast as a, sql_gen as g, sql_ast, sql_rules
from src.data.stage3.compose import _compose_aggregate as compose
from src.data.stage3.compose import Composer, question_order_sql
from src.tokenizer.bpe import Tokenizer

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM orders WHERE order_date >= '2026-01-01'",
    "SELECT AVG(price) FROM items WHERE category = 'kitchen'",
    "SELECT order_date, SUM(quantity) FROM orders GROUP BY order_date HAVING SUM(quantity) >= 10 ORDER BY SUM(quantity) DESC, order_date ASC LIMIT 3",
    "SELECT item_name FROM items WHERE stock > 0 AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')",
    "SELECT MIN(order_date) FROM orders WHERE status = 'shipped'",
])
def test_roundtrip_and_scoring(sql):
    query = sql_ast.parse(sql)
    assert isinstance(query, a.AdvancedQuery)
    assert query.render() == sql
    assert sql_ast.from_dict(sql_ast.to_dict(query)).render() == sql
    assert sql_rules.same_sql(sql, sql)
    assert not sql_rules.select_ambiguous("question", sql)


@pytest.mark.parametrize("sql", [
    "SELECT SUM(name) FROM customers",
    "SELECT status, COUNT(*) FROM orders GROUP BY order_date",
    "SELECT COUNT(*) FROM orders HAVING COUNT(*) > 1",
    "SELECT SUM(quantity) FROM orders GROUP BY status",
    "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM orders)",
    "SELECT COUNT(*) FROM orders WHERE order_date = '2026-02-30'",
])
def test_invalid_structure_rejected(sql):
    with pytest.raises((ValueError, KeyError)):
        sql_ast.parse(sql)


def test_advanced_generation_and_question_semantics(tmp_path):
    g._build_aggregates("data/stage3/db", tmp_path, seed=7, n_train=180, n_eval=32)
    g._verify_aggregates(tmp_path)
    rows = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    mapping = {}
    for record in rows:
        question, trace = compose(record, random.Random(record["sql"]))
        query = a.from_dict(record)
        assert question == compose(record, random.Random(record["sql"]))[0]
        assert all(entry.split == "train" for entry in trace)
        assert sql_rules.label_mismatch(question, query.render()) is False
        previous = mapping.setdefault(question, query.render())
        assert previous == query.render()
        for condition in (*query.conds, *query.inner_conds):
            for value in (condition.val if condition.op == "IN" else [condition.val, condition.val2] if condition.op == "BETWEEN" else [condition.val]):
                if isinstance(value, str):
                    assert value.strip("%") in question
        if query.having:
            assert f"{float(query.having[1]):.6g}" in question
        assert "{" not in question
    g._build_aggregates("data/stage3/db", tmp_path / "repeat", seed=7, n_train=180, n_eval=32)
    for split in ("train", "indist", "struct", "holdout"):
        assert (tmp_path / g.FILES[split]).read_bytes() == (tmp_path / "repeat" / g.FILES[split]).read_bytes()


def test_subquery_inner_and_outer_scope_are_distinct():
    sql = "SELECT item_name FROM items WHERE stock > 0 AND price > (SELECT AVG(price) FROM items WHERE category = 'kitchen')"
    record = a.to_dict(a.parse(sql))
    question, _ = compose(record, random.Random(0))
    assert "kitchen" in question and ("average" in question or "mean" in question)
    assert "subquery" in sql_rules.error_parts(sql, sql.replace("AVG(price)", "MAX(price)"))


@pytest.mark.parametrize("sql", [
    "SELECT SUM(quantity) FROM orders WHERE customer_id = 1452 AND status = 'shipped'",
    "SELECT COUNT(*) FROM customers WHERE name LIKE '%wood%'",
    "SELECT SUM(quantity) FROM orders WHERE customer_id IN (1452, 1453)",
    "SELECT AVG(price) FROM items WHERE item_name IN ('wireless mouse', 'wooden desk') AND stock > 0",
    "SELECT * FROM items WHERE stock > (SELECT AVG(stock) FROM items WHERE item_name IN ('wireless mouse', 'wooden desk'))",
    "SELECT * FROM items WHERE stock > (SELECT AVG(stock) FROM items WHERE item_name LIKE '%desk%' AND category = 'furniture')",
])
def test_reused_filters_work_through_composition_serialization_and_scoring(sql):
    query = sql_ast.parse(sql)
    record = a.to_dict(query)
    assert a.from_dict(json.loads(json.dumps(record))).render() == sql
    comp = Composer.load()
    with sqlite3.connect("data/stage3/db/shop.db") as con:
        con.execute(sql).fetchall()
    for seed in range(10):
        question, _ = compose(record, random.Random(seed), comp)
        label = question_order_sql(record, question)
        assert sql_rules.same_sql(sql, label)
        for c in (*query.conds, *query.inner_conds):
            if c.op == "IN":
                changed = sql_ast.parse(label)
                matching = [c2 for c2 in (*changed.conds, *changed.inner_conds) if c2.col == c.col and c2.op == "IN"]
                assert all([question.index(str(v)) for v in c2.val] == sorted(question.index(str(v)) for v in c2.val)
                           for c2 in matching)


def test_inner_in_holdout_values_cannot_leak():
    query = a.parse("SELECT * FROM items WHERE stock > (SELECT AVG(stock) FROM items WHERE category IN ('kitchen', 'sports'))")
    bans = {"items": {"category": {"sports"}}}
    assert g._aggregate_banned_values(query, bans) == [("category", "sports")]


@pytest.mark.parametrize("sql", [
    "SELECT * FROM items",
    "SELECT * FROM items WHERE item_name IN ('wireless mouse', 'wooden desk') AND stock > 0",
    "SELECT * FROM customers WHERE name = 'vera johnson' OR city = 'boston'",
    "SELECT * FROM customers WHERE city = 'boston' OR membership != 'gold'",
    "SELECT * FROM orders WHERE customer_id = 1452 OR quantity > 5 ORDER BY quantity DESC, order_id ASC LIMIT 3",
])
def test_relaxed_flat_queries_have_working_question_templates(sql):
    record = sql_ast.to_dict(sql_ast.parse(sql))
    composer = Composer.load()
    for seed in range(10):
        question, _ = composer.compose(record, random.Random(seed))
        assert question and "{" not in question
        assert sql_rules.same_sql(sql, question_order_sql(record, question))


def test_empty_and_null_aggregates_are_valid():
    with sqlite3.connect(":memory:") as con:
        con.execute("CREATE TABLE items (item_id INTEGER, stock INTEGER)")
        assert con.execute(a.parse("SELECT COUNT(*) FROM items WHERE item_id = 1").render()).fetchall() == [(0,)]
        assert con.execute(a.parse("SELECT SUM(stock) FROM items WHERE item_id = 1").render()).fetchall() == [(None,)]


def test_sampled_filters_cover_names_ids_and_empty_results():
    from src.data.stage3 import sql_gen
    from src.data import db_gen
    con, ho, bans = sql_gen._open("data/stage3/db")
    try:
        cfg = db_gen.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in ho["config"].items()})
        pools = sql_gen.Pools(con, bans, cfg)
        rng = random.Random(17)
        queries = [q for _ in range(600) if (q := g._sample_aggregate_query(rng, pools)) is not None]
        conditions = [c for q in queries for c in (*q.conds, *q.inner_conds)]
        assert any(c.col in ("name", "item_name") for c in conditions)
        assert any(c.col in ("customer_id", "item_id", "order_id") for c in conditions)
        assert any(c.op == "IN" for c in conditions)
        assert any(c.op == "LIKE" for c in conditions)
        assert any(q.meta["n_rows"] == 0 for q in queries)
        assert any(con.execute(q.render()).fetchall() == [(None,)] for q in queries)
        assert any(con.execute(q.render()).fetchall() == [(0,)] for q in queries if q.aggregate == "COUNT(*)" and not q.group_by)
        assert all(not g._aggregate_banned_values(q, bans) for q in queries)
    finally:
        con.close()


def test_stage3_tokenizer_keywords_and_sql_roundtrip():
    path = Path("data/stage3/tokenizer/tokenizer.json")
    tok = Tokenizer.load(path)
    for word in ("COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP BY", "HAVING"):
        assert len(tok.encode(word)) == 1
    for split in ("train", "indist", "struct", "holdout"):
        records = json.loads((Path("data/stage3/sql") / g.FILES[split]).read_text(encoding="utf-8"))
        for record in records:
            assert tok.decode(tok.encode(record["sql"])) == record["sql"]
