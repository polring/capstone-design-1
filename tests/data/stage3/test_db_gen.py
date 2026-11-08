import json
import sqlite3
from dataclasses import asdict

import pytest

from src.data import db_gen as stage1
from src.data import db_gen
from src.data.stage3.name_pools import load_pools, customer_names, product_names, replacement_pools

pytestmark = pytest.mark.integration


def test_dates_preserve_existing_data_and_holdouts(tmp_path):
    cfg = db_gen.preset_config(3, "large")
    db, ho = db_gen.build(0, str(tmp_path / "new"), cfg)
    old_db, old_ho = stage1.build(0, str(tmp_path / "old"))
    assert db_gen.verify(db, cfg) == []
    with sqlite3.connect(db) as new, sqlite3.connect(old_db) as old:
        for table, cols in stage1.TABLE_COLUMNS.items():
            query = f"SELECT {', '.join(cols)} FROM {table} ORDER BY 1"
            assert new.execute(query).fetchall() == old.execute(query).fetchall()
        assert new.execute("SELECT COUNT(DISTINCT order_date) FROM orders").fetchone()[0] > 1
        assert new.execute("SELECT MAX(n) FROM (SELECT COUNT(*) n FROM orders GROUP BY order_date)").fetchone()[0] > 1
    new_ho = json.loads(open(ho, encoding="utf-8").read())
    original = json.loads(open(old_ho, encoding="utf-8").read())
    for key in ("customers", "items"):
        assert new_ho[key] == original[key]
    assert new_ho["orders"]["heldout_ids"] == original["orders"]["heldout_ids"]
    assert db_gen.test_holdout(ho) == []
    with sqlite3.connect(db) as con:
        observed = {r[0] for r in con.execute("SELECT DISTINCT order_date FROM orders")}
    heldout = set(new_ho["orders"]["heldout_dates"])
    assert len(heldout) == round(len(observed) * cfg.heldout_date_ratio)
    assert db_gen.load_bans(new_ho)["orders"]["order_date"] == heldout
    assert observed - heldout
    _, repeat_ho = db_gen.build(0, str(tmp_path / "repeat"), cfg)
    assert json.loads(open(repeat_ho, encoding="utf-8").read()) == new_ho
    assert db_gen.test_determinism(str(tmp_path / "determinism"), cfg) == []


def test_custom_date_range_and_invalid_range(tmp_path):
    cfg = db_gen.preset_config(3, "large", date_start="2024-02-29", date_end="2024-02-29", heldout_date_ratio=0)
    db, _ = db_gen.build(7, str(tmp_path), cfg)
    assert db_gen.verify(db, cfg) == []
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT DISTINCT order_date FROM orders").fetchall() == [("2024-02-29",)]
    with pytest.raises(ValueError):
        db_gen.preset_config(3, "large", date_start="2026-01-01", date_end="2025-01-01").validate()
    with pytest.raises(ValueError, match="two observed"):
        db_gen.build(0, str(tmp_path / "invalid"), db_gen.preset_config(3, "large",
            date_start="2024-02-29", date_end="2024-02-29"))
    with pytest.raises(ValueError):
        db_gen.preset_config(3, "large", heldout_date_ratio=1).validate()


def test_stage1_database_is_protected():
    with pytest.raises(ValueError, match="stage-1"):
        db_gen.build(0, str(db_gen.config.stage_paths(1).db_dir), db_gen.preset_config(3, "large"))


def test_aggregate_preset_supports_grouped_queries(tmp_path):
    cfg = db_gen.Config(**asdict(db_gen.PRESETS["aggregate"]))
    db, ho = db_gen.build(0, str(tmp_path), cfg)
    assert db_gen.verify(db, cfg) == []
    assert db_gen.test_holdout(ho) == []
    with sqlite3.connect(db) as con:
        for table, expected in (("customers", 600), ("items", 400), ("orders", 6000)):
            assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected
        for col in ("customer_id", "item_id"):
            total = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM orders").fetchone()[0]
            filtered = con.execute(
                f"SELECT {col}, SUM(quantity) FROM orders GROUP BY {col} HAVING COUNT(*) >= 10"
            ).fetchall()
            assert 0 < len(filtered) < total
        dates = con.execute("SELECT COUNT(DISTINCT order_date) FROM orders").fetchone()[0]
        assert dates > 700
    pools = load_pools()
    candidates = replacement_pools(db, pools)
    assert len(customer_names(pools)) >= 20000
    assert sum(map(len, product_names(pools).values())) >= 4000
    with sqlite3.connect(db) as con:
        names = {r[0] for r in con.execute("SELECT name FROM customers")}
        items = {r[0] for r in con.execute("SELECT item_name FROM items")}
    assert not names & set(candidates["name"])
    assert len(candidates["name"]) == len(customer_names(pools)) - cfg.n_customers
    for category, values in candidates["item_name_by_category"].items():
        assert values
        assert not items & set(values)
        assert all(len(n.split()) == 2 for n in values)
        assert set(values) <= set(product_names(pools)[category])
