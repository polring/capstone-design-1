import random

import pytest

from src.data.stage2.synonym_swap import SynonymSwapper

pytestmark = pytest.mark.unit


def pair(question, sql, source="llm:gpt:q1"):
    return {"question": question, "sql": sql, "sql_group": sql, "source": source}


def test_swaps_only_llm_questions_and_keeps_sql():
    """LLM 쌍만 바꾸고 SQL 은 그대로다. 조합기 쌍은 같은 단어가 있어도 건드리지 않는다"""
    sql = "SELECT * FROM items WHERE price < 20.00"
    pairs = [pair("pull up all items under $20", sql), pair("pull up all items under $20", sql, source="composer")]
    out, stats = SynonymSwapper().swap(pairs, 1.0, random.Random(0))
    assert out[0]["synonym_swapped"] and out[0]["question"] in ("bring up all items under $20", "dig up all items under $20")
    assert out[0]["sql"] == sql
    assert not out[1]["synonym_swapped"] and out[1]["question"] == pairs[1]["question"]
    assert stats == {"llm_pairs": 1, "swapped": 1}


def test_keeps_original_when_swap_breaks_value_or_label():
    """값을 바꾸거나(item 이름 catalog → store) 라벨 규칙에 걸리면(find which → get which, L2) 원래 문장을 쓴다"""
    pairs = [pair("show items named catalog", "SELECT * FROM items WHERE item_name = 'catalog'"),
             pair("find which orders have quantity 10", "SELECT order_id FROM orders WHERE quantity = 10")]
    for seed in range(5):
        out, stats = SynonymSwapper().swap(pairs, 1.0, random.Random(seed))
        assert [p["question"] for p in out] == [p["question"] for p in pairs]
        assert stats["rejected"] == 2
