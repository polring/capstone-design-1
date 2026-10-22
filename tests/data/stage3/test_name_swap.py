import random

import pytest

from src.data.stage3 import sql_ast
from src.data.stage3.name_swap import NameSwapper

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def swapper():
    return NameSwapper([])


def swap(swapper, question, sql):
    pair = {"question": question, "sql": sql, "sql_group": sql}
    changed, stats = swapper.swap([pair], 1.0, random.Random(3))
    return pair, changed[0], stats


def test_inner_filter_name_is_replaced(swapper):
    old = next(iter(swapper.product_categories))
    original, changed, stats = swap(
        swapper, f"items with stock above the average stock of {old}",
        f"SELECT * FROM items WHERE stock > (SELECT AVG(stock) FROM items WHERE item_name = '{old}')",
    )
    assert changed["swapped"]
    before, after = sql_ast.parse(original["sql"]), sql_ast.parse(changed["sql"])
    new = after.inner_conds[0].val
    assert new != old and new in changed["question"] and old not in changed["question"]
    assert swapper.product_categories[new] == swapper.product_categories[old]
    assert before.conds == after.conds and before.subquery == after.subquery
    assert changed["sql_group"] == original["sql_group"]
    assert stats["swapped_inner_conditions"] == 1
    assert sql_ast.violations(after) == []


@pytest.mark.parametrize("repeat_question", [False, True])
def test_same_name_in_both_scopes_has_one_consistent_replacement(swapper, repeat_question):
    old = next(iter(swapper.product_categories))
    question = f"show {old} with stock above its average stock"
    if repeat_question:
        question = f"show {old} with stock above the average stock of {old}"
    _, changed, stats = swap(
        swapper, question,
        f"SELECT * FROM items WHERE item_name = '{old}' AND stock > "
        f"(SELECT AVG(stock) FROM items WHERE item_name = '{old}')",
    )
    assert changed["swapped"]
    query = sql_ast.parse(changed["sql"])
    assert query.conds[0].val == query.inner_conds[0].val
    assert changed["question"].count(query.conds[0].val) == (2 if repeat_question else 1)
    assert stats["swapped_item_name_eq"] == 2


def test_different_names_across_scopes_stay_distinct(swapper):
    old, other = list(swapper.product_categories)[:2]
    _, changed, _ = swap(
        swapper, f"show {old} with stock above the average stock of {other}",
        f"SELECT * FROM items WHERE item_name = '{old}' AND stock > "
        f"(SELECT AVG(stock) FROM items WHERE item_name = '{other}')",
    )
    assert changed["swapped"]
    query = sql_ast.parse(changed["sql"])
    assert query.conds[0].val != query.inner_conds[0].val
    assert query.conds[0].val in changed["question"]
    assert query.inner_conds[0].val in changed["question"]


def test_aggregate_group_having_and_order_are_preserved(swapper):
    old = next(iter(swapper.product_categories))
    original, changed, _ = swap(
        swapper, f"group {old} by category and sum stock, keep the top 3",
        f"SELECT category, SUM(stock) FROM items WHERE item_name = '{old}' "
        "GROUP BY category HAVING SUM(stock) > 15 ORDER BY SUM(stock) DESC, category ASC LIMIT 3",
    )
    assert changed["swapped"]
    before, after = sql_ast.parse(original["sql"]), sql_ast.parse(changed["sql"])
    for field in ("aggregate", "group_by", "having", "order", "limit"):
        assert getattr(before, field) == getattr(after, field)


def test_missing_inner_name_rolls_back_entire_pair(swapper):
    old, other = list(swapper.product_categories)[:2]
    original, changed, stats = swap(
        swapper, f"show {old} above a reference average",
        f"SELECT * FROM items WHERE item_name = '{old}' AND stock > "
        f"(SELECT AVG(stock) FROM items WHERE item_name = '{other}')",
    )
    assert changed == dict(original, swapped=False)
    assert stats["skipped"] == 1


def test_ambiguous_repeated_mention_in_single_scope_is_skipped(swapper):
    old = next(iter(swapper.product_categories))
    original, changed, _ = swap(swapper, f"show {old}, i mean {old}",
                                f"SELECT * FROM items WHERE item_name = '{old}'")
    assert changed == dict(original, swapped=False)


def test_quoted_common_word_like_stays_intact(swapper):
    _, changed, _ = swap(
        swapper, "items in stock with names ending in 'in'",
        "SELECT * FROM items WHERE item_name LIKE '%in' AND stock > 0",
    )
    assert changed["swapped"]
    fragment = sql_ast.parse(changed["sql"]).conds[0].val.strip("%")
    assert len(fragment) == 2 and fragment != "in"
    assert changed["question"] == f"items in stock with names ending in '{fragment}'"


@pytest.fixture(scope="module")
def balanced_swapper():
    from src.data.stage3.like_sampling import LikeFragmentSampler
    from src.tokenizer.bpe import Tokenizer
    sampler = LikeFragmentSampler([], Tokenizer([]), banned_fragments={"zzz"})
    return NameSwapper([], like_sampler=sampler)


@pytest.mark.parametrize("pattern", ["abc%", "%abc", "%abc%"])
def test_balanced_like_preserves_direction_conditions_and_metadata(balanced_swapper, pattern):
    """균형 LIKE 교체도 방향·길이·다른 조건·출력·학습 분할 메타데이터를 보존한다"""
    pair = {"question": "show item names matching 'abc' with stock at least 5",
            "sql": f"SELECT item_name FROM items WHERE item_name LIKE '{pattern}' AND stock >= 5",
            "sql_group": "original-group", "round": "r", "contrast_group": "g"}
    result, stats = balanced_swapper.swap([pair], 1.0, random.Random(4))
    after = result[0]
    query = sql_ast.parse(after["sql"])
    fragment = query.conds[0].val.strip("%")
    assert len(fragment) == 3 and fragment not in {"abc", "zzz"}
    assert query.conds[0].val == pattern.replace("abc", fragment)
    assert after["question"] == pair["question"].replace("abc", fragment)
    assert query.conds[1] == sql_ast.parse(pair["sql"]).conds[1]
    assert after["orig_sql"] == pair["sql"] and pair["question"] == after["orig_question"]
    assert all(after[k] == pair[k] for k in ("sql_group", "round", "contrast_group"))
    assert sum(v for k, v in stats.items() if k.startswith("like_bucket_")) == 1


@pytest.mark.parametrize("repeat", [False, True])
def test_balanced_like_reuses_one_value_across_both_scopes(balanced_swapper, repeat):
    """동일 LIKE가 안팎에 있어도 일관된 값 하나로 교체하고 출현 횟수를 보존한다"""
    question = "items containing 'abc' with stock above their average"
    if repeat:
        question = "items containing 'abc' with stock above the average for items containing 'abc'"
    pair = {"question": question, "sql": "SELECT * FROM items WHERE item_name LIKE '%abc%' AND stock > "
            "(SELECT AVG(stock) FROM items WHERE item_name LIKE '%abc%')"}
    changed, stats = balanced_swapper.swap([pair], 1.0, random.Random(2))
    query = sql_ast.parse(changed[0]["sql"])
    assert changed[0]["swapped"]
    assert query.conds[0].val == query.inner_conds[0].val
    assert changed[0]["question"].count(query.conds[0].val.strip("%")) == (2 if repeat else 1)
    assert sum(v for k, v in stats.items() if k.startswith("like_bucket_")) == 1


def test_balanced_quoted_word_and_unsafe_pair_rollback(balanced_swapper):
    """따옴표 밖의 일반 단어는 유지하고 교체 대상을 찾지 못하면 전체를 되돌린다"""
    pair = {"question": "items in stock ending in 'in'", "sql":
            "SELECT * FROM items WHERE item_name LIKE '%in' AND stock > 0"}
    changed, _ = balanced_swapper.swap([pair], 1.0, random.Random(3))
    fragment = sql_ast.parse(changed[0]["sql"]).conds[0].val.strip("%")
    assert changed[0]["question"] == f"items in stock ending in '{fragment}'"
    unsafe = {"question": "show inventory", "sql": "SELECT * FROM items WHERE item_name LIKE '%abc%'"}
    changed, stats = balanced_swapper.swap([unsafe], 1.0, random.Random(3))
    assert changed == [dict(unsafe, swapped=False)]
    assert stats["skipped"] == 1 and not any(k.startswith("like_bucket_") for k in stats)
