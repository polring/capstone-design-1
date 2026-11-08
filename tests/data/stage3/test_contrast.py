import random
import pytest

from src.data.stage3 import compose, contrast, sql_ast
from src.data.stage3.sql_ast import Cond, Query


@pytest.mark.unit
def test_comparison_contrast_changes_only_one_operator():
    q = Query("items", "item_name", (Cond("price", ">", 30.0), Cond("stock", ">", 10)), "AND")
    rows = contrast.variants(q, "comparison", random.Random(4))
    assert len(rows) == 5
    assert {c.op for row in rows for c in row.conds} >= {"=", "<", ">", "<=", ">="}
    # Distinct thresholds or output columns must not accidentally explain the label.
    assert len({tuple((c.col, c.val) for c in row.conds) for row in rows}) == 1
    assert {row.select for row in rows} == {"item_name"}
    assert all(not sql_ast.violations(sql_ast.canonical(row)) for row in rows)


@pytest.mark.unit
def test_like_contrast_preserves_fragment_and_other_filters():
    q = Query("customers", "customer_id", (Cond("name", "LIKE", "ab%"), Cond("city", "=", "boston")), "AND")
    rows = contrast.variants(q, "like", random.Random(0))
    assert [row.conds[0].val for row in rows] == ["ab%", "%ab", "%ab%"]
    assert all(row.conds[1:] == q.conds[1:] and row.select == q.select for row in rows)


@pytest.mark.unit
def test_order_contrast_isolates_direction_column_and_count():
    q = Query("items", "item_name", (Cond("category", "=", "kitchen"),))
    a, b, c, d = contrast.variants(q, "ordering", random.Random(0))
    assert a.order[0] == b.order[0] and a.order[1] != b.order[1]
    assert a.limit == b.limit == c.limit
    assert a.order[0] != c.order[0] and a.order == d.order and a.limit != d.limit
    assert all(row.conds == q.conds for row in (a, b, c, d))


@pytest.mark.integration
def test_composer_keeps_every_arm_rule_consistent_and_train_only():
    comp = compose.Composer.load("train")
    q = Query("items", "item_id", (Cond("item_name", "LIKE", "ab%"),))
    queries = [sql_ast.canonical(x) for x in contrast.variants(q, "like", random.Random(0))]
    rows = comp.compose_contrast(queries, 0, "test_like")
    assert len(rows) == 3 and all(len(arm) == 5 for arm in rows)
    for query, arm in zip(queries, rows):
        assert len({r["question"] for r in arm}) == 5
        assert all(not compose.drop_reason(r) for r in arm)
        assert all(sql_ast.parse(r["sql"]).render() == query.render() for r in arm)
        assert all("eval" not in ref.split(":") for r in arm for ref in r["template_refs"])
    assert rows == comp.compose_contrast(queries, 0, "test_like")
    with pytest.raises(ValueError):
        compose.Composer.load("eval").compose_contrast(queries, 0, "test_like")


@pytest.mark.unit
def test_banned_like_direction_cannot_bypass_fragment_holdout():
    bans = {"items": {"item_name": {"xy"}}}
    q = Query("items", "item_id", (Cond("item_name", "LIKE", "xy%"),))
    assert all(contrast.banned_values(row, bans)
               for row in contrast.variants(q, "like", random.Random(0)))
