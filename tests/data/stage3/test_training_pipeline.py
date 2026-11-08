import json
import random
import sqlite3
from pathlib import Path

import pytest
import torch

from src.data.stage3 import sql_ast, sql_rules
from src.data.stage3.name_swap import NameSwapper
from src.data.dataset import load_train_pairs
from src.evaluation.scoring import execute
from src.inference.generate import generate_batch
from src.evaluation.eval_sets import list_sets, load_eval_set

pytestmark = pytest.mark.integration


def test_two_word_swap_is_db_disjoint_and_preserves_sql():
    pairs = [p for p in load_train_pairs(train_dir="data/stage3/train")
             if p["source"].startswith("composer")]
    swapper = NameSwapper(pairs)
    sample = random.Random(7).sample(pairs, 5000)
    swapped, stats = swapper.swap(sample, 1, random.Random(0))
    assert stats["swapped_pairs"] > 0
    with sqlite3.connect("data/stage3/db/shop.db") as con:
        db_names = {r[0] for r in con.execute("SELECT name FROM customers")}
        db_names |= {r[0] for r in con.execute("SELECT item_name FROM items")}
    for original, changed in zip(sample, swapped):
        if not changed["swapped"]:
            continue
        old = sql_ast.parse(original["sql"])
        new = sql_ast.parse(changed["sql"])
        assert sql_ast.violations(sql_ast.canonical(new)) == []
        assert old.order == new.order and old.limit == new.limit
        assert original["sql_group"] == changed["sql_group"]
        for before, after in zip(old.conds, new.conds):
            if before.col not in ("name", "item_name"):
                assert before == after
            elif before.op in ("=", "IN"):
                values = after.val if after.op == "IN" else [after.val]
                assert len(values) == len(set(values))
                assert all(len(v.split()) == 2 and v not in db_names and v in changed["question"] for v in values)
                if before.col == "item_name":
                    olds = before.val if before.op == "IN" else [before.val]
                    assert all(swapper.product_categories[a] == swapper.product_categories[b]
                               for a, b in zip(olds, values))


def test_eval_sets_and_date_rules():
    directory = "data/stage3/eval"
    names = set(list_sets(directory))
    assert {"indist", "struct", "holdout", "phrase"} <= names
    assert names <= {"indist", "struct", "holdout", "phrase", "advanced_indist", "advanced_struct",
                     "advanced_holdout", "advanced_phrase", "ext_r1_gpt_indist", "ext_r1_gpt_struct",
                     "ext_r1_gpt_holdout", "ext_r2_humanstyle_gpt_indist",
                     "ext_r2_humanstyle_gpt_struct", "ext_r2_humanstyle_gpt_holdout"}
    for name in list_sets(directory):
        pairs, meta = load_eval_set(name, directory)
        assert pairs and meta
        for pair in pairs:
            assert sql_rules.parse_sql(pair["sql"]) is not None
            # Composer sentences follow the checker; frozen external questions
            # retain their original labels and are reported separately.
            if not name.startswith("ext_"):
                assert not sql_rules.label_mismatch(pair["question"], pair["sql"])


def test_execution_accuracy_preserves_order():
    with sqlite3.connect(":memory:") as con:
        con.execute("CREATE TABLE values_table (n INTEGER)")
        con.executemany("INSERT INTO values_table VALUES (?)", [(1,), (2,)])
        assert execute(con, "SELECT n FROM values_table ORDER BY n ASC", ordered=True) != execute(
            con, "SELECT n FROM values_table ORDER BY n DESC", ordered=True)


def test_generation_supports_more_than_64_sql_tokens():
    class LongOutput(torch.nn.Module):
        max_seq_len = 256

        def forward(self, ids):
            logits = torch.zeros(ids.shape[0], ids.shape[1], 260)
            logits[:, -1, 258 if ids.shape[1] >= 85 else 65] = 1
            return logits

    result = generate_batch(LongOutput(), [{"ids": [257], "sql_start": 1}],
                            torch.device("cpu"), None)
    assert result == [[65] * 84]
