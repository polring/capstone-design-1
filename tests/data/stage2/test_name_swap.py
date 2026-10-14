import random
import re
from pathlib import Path

import pytest

from src.data import db_gen
from src.data.stage2 import name_swap as ns
from src.data.stage2 import sql_ast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
A, B, C = db_gen.GIVEN_NAMES[:3]

IN_PAIR = {"question": f"where do the customers who go by {A}, {B} or {C} live?",
           "sql": f"SELECT city FROM customers WHERE name IN ('{A}', '{B}', '{C}')"}
LIKE_PAIR = {"question": "show the members based in seattle, with 'dwa' in their names",
             "sql": "SELECT * FROM customers WHERE name LIKE '%dwa%' AND city = 'seattle'"}
QUOTED_PAIR = {"question": "find the cities of the members in boston with names that end in 'in'",
               "sql": "SELECT city FROM customers WHERE name LIKE '%in' AND city = 'boston'"}
STAGE1_PAIR = {"question": f"what city does {A} live in?", "sql": f"SELECT city FROM customers WHERE name = '{A}'"}
NO_NAME_PAIR = {"question": "items under $20", "sql": "SELECT * FROM items WHERE price < 20.00"}
TWICE_PAIR = {"question": f"is {A} there? i mean {A}", "sql": f"SELECT name FROM customers WHERE name = '{A}'"}


@pytest.fixture
def swapper(monkeypatch):
    # NameSwapper 는 holdout.json 을 cwd 기준 상대 경로로 읽는다
    monkeypatch.chdir(PROJECT_ROOT)
    return ns.NameSwapper([IN_PAIR, LIKE_PAIR, QUOTED_PAIR, STAGE1_PAIR, NO_NAME_PAIR], english=set())


def swap_one(swapper, pair, seed=0):
    out, _ = swapper.swap([pair], 1.0, random.Random(seed))
    return out[0]


@pytest.mark.unit
def test_in_swaps_every_name_in_place(swapper):
    """IN 의 이름 전부가 서로 다른 가짜 이름으로 바뀌고, SQL 은 다시 정렬하지 않는다 (값 자리 그대로)"""
    o = swap_one(swapper, IN_PAIR)
    assert o["swapped"] and o["orig_sql"] == IN_PAIR["sql"]
    q = sql_ast.parse(o["sql"])
    news = list(q.conds[0].val)
    assert len(set(news)) == 3 and not {A, B, C} & set(news)
    assert o["sql"] == "SELECT city FROM customers WHERE name IN (" + ", ".join(f"'{n}'" for n in news) + ")"
    assert all(re.search(rf"\b{n}\b", o["question"]) for n in news)
    assert not any(re.search(rf"\b{n}\b", o["question"]) for n in (A, B, C))


@pytest.mark.unit
def test_in_keeps_question_order(swapper):
    """질문에 나온 순서 = SQL 값 순서인 쌍은 바꾼 뒤에도 그 관계가 유지된다 (질문 순서가 값 순서와 달라도)"""
    reordered = {"question": f"where do the customers who go by {C}, {A} or {B} live?",
                 "sql": f"SELECT city FROM customers WHERE name IN ('{C}', '{A}', '{B}')"}
    for seed, pair in ((s, p) for s in range(20) for p in (IN_PAIR, reordered)):
        o = swap_one(swapper, pair, seed)
        news = list(sql_ast.parse(o["sql"]).conds[0].val)
        pos = [o["question"].index(n) for n in news]
        assert pos == sorted(pos)


@pytest.mark.unit
def test_like_fragment_keeps_mode_and_length(swapper):
    """LIKE 조각은 같은 방향(접두·접미·포함)·같은 길이의 가짜 조각으로 바뀌고 질문의 조각도 같이 바뀐다"""
    o = swap_one(swapper, LIKE_PAIR)
    cond = sql_ast.parse(o["sql"]).conds[0]
    assert cond.col == "name" and cond.op == "LIKE"
    frag = cond.val.strip("%")
    assert cond.val == f"%{frag}%" and len(frag) == 3 and frag != "dwa"
    assert o["question"] == LIKE_PAIR["question"].replace("dwa", frag)
    assert "city = 'seattle'" in o["sql"]


@pytest.mark.unit
def test_fragment_that_is_a_common_word_swaps_only_quoted(swapper):
    """조각 "in" 이 질문의 다른 단어 in 과 겹치면 따옴표 안의 것만 바꾼다"""
    for seed in range(20):
        o = swap_one(swapper, QUOTED_PAIR, seed)
        assert o["swapped"]
        frag = sql_ast.parse(o["sql"]).conds[0].val.strip("%")
        assert o["question"] == QUOTED_PAIR["question"].replace("'in'", f"'{frag}'")
        assert o["sql"] == f"SELECT city FROM customers WHERE name LIKE '%{frag}' AND city = 'boston'"


@pytest.mark.unit
def test_stage1_pair_and_pairs_without_names(swapper):
    """1단계 모양 쌍도 바뀌고, 이름 조건이 없는 쌍은 그대로다"""
    out, stats = swapper.swap([STAGE1_PAIR, NO_NAME_PAIR], 1.0, random.Random(0))
    assert out[0]["swapped"] and A not in out[0]["question"]
    assert out[1] == dict(NO_NAME_PAIR, swapped=False)
    assert stats["name_pairs"] == 1 and stats["total_pairs"] == 2


@pytest.mark.unit
def test_skip_when_value_appears_more_than_once(swapper):
    """바꿀 값이 질문에 두 번 이상 나오면 어느 것을 바꿀지 정할 수 없어 바꾸지 않는다"""
    out, stats = swapper.swap([TWICE_PAIR], 1.0, random.Random(0))
    assert out[0] == dict(TWICE_PAIR, swapped=False) and stats["skipped"] == 1


@pytest.mark.unit
def test_ratio_and_determinism(swapper):
    """ratio 0 이면 바꾸지 않고, 같은 seed 면 결과가 같다"""
    pairs = [IN_PAIR, LIKE_PAIR, STAGE1_PAIR] * 20
    none, _ = swapper.swap(pairs, 0.0, random.Random(0))
    assert not any(p["swapped"] for p in none)
    a, _ = swapper.swap(pairs, 0.5, random.Random(3))
    b, _ = swapper.swap(pairs, 0.5, random.Random(3))
    assert a == b and 0 < sum(p["swapped"] for p in a) < len(pairs)
