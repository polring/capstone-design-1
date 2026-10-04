import random
import zipfile
from pathlib import Path

import pytest

from src.data import name_swap as ns
from src.data.generator import db_gen
from src.data.generator.question_gen import literal_in_question

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUSTOMER = db_gen.GIVEN_NAMES[0]
ITEM = next(iter(db_gen.CATEGORY_NOUNS.values()))[0]


def make_pairs() -> list[dict]:
    return [
        {"question": f"what city does {CUSTOMER} live in?",
         "sql": f"SELECT city FROM customers WHERE name = '{CUSTOMER}'"},
        {"question": f"how many {ITEM}s are in stock?",
         "sql": f"SELECT stock FROM items WHERE item_name = '{ITEM}'"},
        {"question": f"is {CUSTOMER}'s membership gold?",
         "sql": f"SELECT membership FROM customers WHERE name = '{CUSTOMER}'"},
        {"question": "what's the price of item 3155?",
         "sql": "SELECT price FROM items WHERE item_id = 3155"},
    ]


@pytest.fixture
def swapper(monkeypatch):
    # NameSwapper 는 holdout.json 을 cwd 기준 상대 경로로 읽는다
    monkeypatch.chdir(PROJECT_ROOT)
    return ns.NameSwapper(make_pairs(), english=set())


# ===========================================================================
# MarkovNameGenerator
# ===========================================================================

@pytest.mark.unit
def test_markov_deterministic_and_within_alphabet():
    """같은 seed 면 같은 이름, 글자는 학습 이름의 알파벳 안, 길이는 [min_len, max_len]"""
    names = ["anna", "hannah", "joanna", "nathan", "hank"]
    gen = ns.MarkovNameGenerator(names)
    a = [gen.sample(random.Random(1)) for _ in range(5)]
    b = [gen.sample(random.Random(1)) for _ in range(5)]
    assert a == b
    alphabet = set("".join(names))
    rng = random.Random(0)
    samples = [s for s in (gen.sample(rng, min_len=3, max_len=8) for _ in range(200)) if s]
    assert samples
    assert all(set(s) <= alphabet and 3 <= len(s) <= 9 for s in samples)


# ===========================================================================
# 영단어 판정 / 목록 읽기
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("word, expected", [
    ("dog", True), ("dogs", True), ("parked", True), ("parking", True), ("stories", True),
    ("baked", True), ("quickly", True), ("zorbak", False), ("dogx", False),
])
def test_is_english(word, expected):
    """표제어 및 흔한 변화형(-s/-es/-ies/-ed/-ing/-er/-ly)을 영단어로 판정"""
    english = {"dog", "park", "story", "bake", "quick"}
    assert ns.is_english(word, english) is expected


@pytest.mark.unit
def test_load_english_words_text_file(tmp_path):
    """한 줄에 한 단어 텍스트 파일: 소문자화, 빈 줄 무시"""
    f = tmp_path / "words.txt"
    f.write_text("Apple\n\nbanana\n  Cherry \n", encoding="utf-8")
    assert ns.load_english_words(f) == {"apple", "banana", "cherry"}


@pytest.mark.unit
def test_load_english_words_wordnet_zip(tmp_path):
    """WordNet zip: index.noun/verb/adj/adv 의 표제어만, 라이선스 헤더(공백 시작)·비알파벳 제외"""
    z = tmp_path / "wordnet.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("wordnet/index.noun", "  1 license header\ndog n 1\nice_cream n 1\nCat n 1\n")
        zf.writestr("wordnet/index.verb", "run v 2\n")
        zf.writestr("wordnet/data.noun", "ignored x\n")
    assert ns.load_english_words(z) == {"dog", "cat", "run"}


@pytest.mark.unit
def test_load_english_words_missing(tmp_path):
    """목록 파일이 없으면 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        ns.load_english_words(tmp_path / "nope.zip")


# ===========================================================================
# replace_name
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("question, expected", [
    ("where does ana live?", "where does zorb live?"),
    ("is ana's order shipped?", "is zorb's order shipped?"),
    ("how many anas are there", "how many zorbs are there"),
    ("is anastasia here?", "is anastasia here?"),  # 다른 단어 안에 묻힌 것은 바꾸지 않음
])
def test_replace_name(question, expected):
    """단어 경계 기준으로 이름만 바꾸고 뒤의 's/s/es 는 유지"""
    assert ns.replace_name(question, "ana", "zorb") == expected


# ===========================================================================
# NameSwapper
# ===========================================================================

@pytest.mark.unit
def test_swap_ratio_zero_keeps_pairs(swapper):
    """ratio 0 이면 아무 쌍도 바뀌지 않음"""
    pairs = make_pairs()
    out, stats = swapper.swap(pairs, 0.0, random.Random(0))
    assert [{k: p[k] for k in ("question", "sql")} for p in out] == pairs
    assert not any(p["swapped"] for p in out)
    assert stats["name_pairs"] == 3 and stats["total_pairs"] == 4


@pytest.mark.unit
def test_swap_ratio_one_swaps_question_and_sql_consistently(swapper):
    """ratio 1 이면 이름 조건 쌍 전부를 질문·SQL 양쪽에서 같은 새 이름으로 바꾸고, 원본을 남김"""
    pairs = make_pairs()
    out, stats = swapper.swap(pairs, 1.0, random.Random(0))
    assert stats.get("swapped_name", 0) + stats.get("swapped_item_name", 0) + stats.get("skipped", 0) == 3

    new_names = []
    for orig, p in zip(pairs, out):
        m = ns.NAME_SQL.search(orig["sql"])
        if not m:
            assert p["swapped"] is False and p["sql"] == orig["sql"]
            continue
        if not p["swapped"]:
            continue
        new = ns.NAME_SQL.search(p["sql"]).group(2)
        new_names.append(new)
        assert p["orig_question"] == orig["question"] and p["orig_sql"] == orig["sql"]
        assert literal_in_question(p["question"], new, "text")
        assert not literal_in_question(p["question"], m.group(2), "text")
        assert p["sql"] == orig["sql"].replace(f"'{m.group(2)}'", f"'{new}'")
    assert new_names, "적어도 하나는 바뀌어야 함"
    assert len(set(new_names)) == len(new_names), "한 번 호출 안에서는 쌍마다 서로 다른 이름"


@pytest.mark.unit
def test_swap_keeps_plural_suffix(swapper):
    """'mouses' 같은 복수형 질문에서 이름만 바꾸고 어미는 유지"""
    out, _ = swapper.swap(make_pairs()[1:2], 1.0, random.Random(0))
    p = out[0]
    assert p["swapped"]
    new = ns.NAME_SQL.search(p["sql"]).group(2)
    assert p["question"] == f"how many {new}s are in stock?"


@pytest.mark.unit
def test_swap_names_avoid_banned_words(swapper):
    """가짜 이름은 실제 이름 풀·holdout 이름·학습 질문 단어와 겹치지 않아야 함"""
    pairs = make_pairs() * 30
    out, _ = swapper.swap(pairs, 1.0, random.Random(0))
    new_names = {ns.NAME_SQL.search(p["sql"]).group(2) for p in out if p["swapped"]}
    assert new_names and not new_names & swapper.base_banned


@pytest.mark.unit
def test_swap_rejects_english_words(monkeypatch):
    """영단어 목록에 있는 후보는 버리고 rejected_english 로 센다"""
    monkeypatch.chdir(PROJECT_ROOT)
    swapper = ns.NameSwapper(make_pairs(), english=set())
    out, stats = swapper.swap(make_pairs()[:1], 1.0, random.Random(0))
    assert out[0]["swapped"] and stats.get("rejected_english", 0) == 0
    first = ns.NAME_SQL.search(out[0]["sql"]).group(2)

    # 같은 seed 로 다시 돌리면 같은 후보가 먼저 나오므로, 그 후보를 영단어로 만들면 반드시 거절된다
    swapper.english = {first}
    out2, stats2 = swapper.swap(make_pairs()[:1], 1.0, random.Random(0))
    assert stats2["rejected_english"] >= 1
    assert ns.NAME_SQL.search(out2[0]["sql"]).group(2) != first


@pytest.mark.unit
def test_swap_deterministic(swapper):
    """같은 rng seed 면 같은 결과"""
    a, _ = swapper.swap(make_pairs(), 0.5, random.Random(7))
    b, _ = swapper.swap(make_pairs(), 0.5, random.Random(7))
    assert a == b
