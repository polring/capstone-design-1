import random
from collections import Counter

import pytest

from src.data.stage3.compose import choose_styled, infer_style

pytestmark=pytest.mark.unit


def test_style_probability_is_independent_of_library_bucket_size():
    entries=[('common',i) for i in range(100)]+[('conversational',0),('contextual',0)]
    rng=random.Random(24)
    counts=Counter(choose_styled(entries,rng,lambda x:x[0])[0] for _ in range(10000))
    for style,expected in [('common',.7),('conversational',.2),('contextual',.1)]:
        assert abs(counts[style]/10000-expected)<.02


def test_missing_style_is_renormalized_without_inventing_templates():
    entries=[('common','a'),('conversational','b')]
    rng=random.Random(24)
    counts=Counter(choose_styled(entries,rng,lambda x:x[0])[0] for _ in range(10000))
    assert counts['contextual']==0
    assert abs(counts['common']/10000-7/9)<.02


@pytest.mark.parametrize('text,style',[
    ('show {rows}','common'),('what is the {aggregate}?','common'),
    ('could you give me {projection}?','conversational'),
    ('i am reviewing {rows}; give me {aggregate}','contextual')])
def test_legacy_style_classification(text,style):
    assert infer_style(text)==style
