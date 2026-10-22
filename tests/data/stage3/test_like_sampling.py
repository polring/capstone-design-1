import random
from collections import Counter

import pytest

from src.data.stage3.like_sampling import LikeFragmentSampler
from src.tokenizer.bpe import Tokenizer

pytestmark = pytest.mark.unit


@pytest.fixture
def sampler():
    # Token 260 (ab) is common, token 261 (xy) is rare; zz is two bytes.
    tok = Tokenizer([], [((97, 98), 260), ((120, 121), 261)])
    common = {"sql": "SELECT name FROM customers WHERE name LIKE '%ab%'"}
    rare = {"sql": "SELECT name FROM customers WHERE name LIKE '%xy%'", "question": "ab " * 100}
    return LikeFragmentSampler([common] * 3 + [rare], tok, rare_max_count=1,
                               banned_fragments={"qq", "rr"})


def test_buckets_use_training_sql_frequency_not_token_id_or_question(sampler):
    """빈도는 학습 SQL의 실제 출현 수이며 토큰 번호나 질문의 빈도를 쓰지 않는다"""
    pools = sampler.pools(2)
    assert pools["common"] == ("ab",)
    assert pools["rare"] == ("xy",)
    assert sampler.token_counts[260] == 3 and sampler.token_counts[261] == 1
    assert "zz" in pools["multi"]
    assert not {"qq", "rr"} & set().union(*map(set, pools.values()))


def test_available_buckets_are_balanced_and_sampling_is_repeatable(sampler):
    """크기가 다른 세 후보군도 동일 확률로 선택하고 같은 seed에서 재현한다"""
    def draw(seed):
        rng = random.Random(seed)
        return [sampler.sample(2, "items containing 'jo'", set(), rng) for _ in range(3000)]
    samples = draw(7)
    assert samples == draw(7)
    counts = Counter(bucket for _, bucket in samples)
    assert set(counts) == {"common", "rare", "multi"}
    assert all(850 < n < 1150 for n in counts.values())


def test_existing_words_used_values_and_banned_literals_are_never_selected(sampler):
    """기존 단어·다른 교체값·금지 리터럴을 피하고 사용 가능한 후보군에서 고른다"""
    rng = random.Random(3)
    for _ in range(100):
        fragment, bucket = sampler.sample(2, "AB and jo", {"xy", "zz"}, rng)
        assert fragment not in {"ab", "jo", "xy", "zz", "qq", "rr"}
        assert bucket == "multi"
    assert sampler.sample(4, "longer fragment", set(), rng) is None
    all_candidates = set().union(*map(set, sampler.pools(2).values()))
    assert sampler.sample(2, "", all_candidates, rng) is None


def test_single_character_sampling_does_not_require_a_multi_token_bucket(sampler):
    """한 글자는 다중 토큰 후보군이 없어도 정상 선택한다"""
    fragment, bucket = sampler.sample(1, "a b c", set(), random.Random(1))
    assert len(fragment) == 1 and fragment not in "abc"
    assert bucket in {"common", "rare"}
