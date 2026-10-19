"""조합 하한(ensure_pair_floor): 특징 쌍마다 하한을 채우되 학습 수와 다른 제약을 지킨다."""
import random

import pytest

from src.data.stage3 import sql_gen as g

pytestmark = pytest.mark.unit


def features(q):
    return {("a", q[0]), ("b", q[1])}


def test_rare_pair_is_filled_by_swapping_redundant_rows():
    training = [("x", "y")] * 80 + [("x", "z")] * 5
    pool = random.Random(1)

    def sample(pair):
        return ("x", "z") if pool.random() < .5 else ("x", "y")

    report = g.ensure_pair_floor(training, features, sample, 20, random.Random(0))
    assert len(training) == 85
    assert training.count(("x", "z")) == 20
    assert report["n_short"] == 0 and report["swapped"] == 15


def test_unreachable_pairs_are_reported_and_constraints_hold():
    training = [("x", "y")] * 30 + [("w", "z")] * 2
    protected = ("x", "y")

    def removable(old, new):
        return training.count(protected) > 28 or old != protected

    report = g.ensure_pair_floor(training, features, lambda pair: None, 10, random.Random(0), removable,
                                 attempts_per_pair=5)
    assert report["short"] == {"a=w | b=z": 2}
    assert training.count(("w", "z")) == 2


def test_same_stratum_is_replaced_first():
    training = [("p", "y")] * 40 + [("q", "y")] * 40 + [("p", "rare")]

    def sample(pair):
        return ("p", "rare")

    g.ensure_pair_floor(training, features, sample, 10, random.Random(0), stratum=lambda q: q[0])
    assert training.count(("q", "y")) == 40
    assert training.count(("p", "rare")) == 10


def test_feature_pairs_of_real_queries():
    from src.data.stage3 import sql_ast

    plain = sql_ast.parse("SELECT * FROM items WHERE category = 'kitchen' AND price < 20.00")
    pairs = g.feature_pairs(g.plain_features(plain))
    assert (("cond", "cat:="), ("cond", "num:cmp")) in pairs
    advanced = sql_ast.parse("SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items WHERE stock > 5)")
    feats = g.aggregate_features(advanced)
    assert {("fam", "scalar_subquery"), ("icond", "num:cmp"), ("sfn", "AVG"), ("proj", "ident")} <= feats
