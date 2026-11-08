import json
import pytest
from collections import Counter, defaultdict

from src.data.preprocessing.wordnet_map import WordNetGraph, nearest, train_vocabulary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("compressed", [False, True])
def test_training_vocabulary_excludes_entities_and_numbers(tmp_path, compressed):
    import gzip

    p = tmp_path / ('train.json.gz' if compressed else 'train.json')
    opener = gzip.open if compressed else open
    with opener(p, 'wt', encoding='utf-8') as stream:
        json.dump([dict(question="show articles named silver patrons in new york with stock 25",
            sql="SELECT * FROM items WHERE item_name = 'silver patrons' AND category = 'new york' AND stock = 25")], stream)
    counts, total = train_vocabulary([p])
    assert total == 1
    assert counts['articles'] == 1 and counts['stock'] == 1
    assert not any(w in counts for w in ['silver','patrons','new','york','25'])


def graph():
    g = WordNetGraph.__new__(WordNetGraph)
    g.words = {p:defaultdict(set) for p in ['n','v','a','r']}
    g.words['n']['customer'] = {'n:1'}
    g.words['n']['client'] = {'n:1'}
    g.words['n']['person'] = {'n:2'}
    g.words['n']['stranger'] = {'n:3'}
    g.exceptions = {p:{} for p in g.words}
    g.edges = defaultdict(set, {'n:1':{'n:2'},'n:2':{'n:1','n:3'},'n:3':{'n:2'}})
    return g


def test_shortest_distance_then_training_frequency_and_fixed_cutoff():
    g = graph()
    counts = Counter(customer=10, client=30, person=2)
    result = nearest(g, counts, 2, 'singular')
    assert result['n:1'][:3] == (0,-30,'client')
    assert result['n:3'][:3] == (1,-2,'person')  # closer beats more frequent
    assert 'n:3' not in nearest(g, counts, 0, 'singular')


def test_noun_number_is_matched_without_pos_tagger():
    g = graph()
    counts = Counter(customer=100, clients=2)
    plural = nearest(g, counts, 0, 'plural')
    assert plural['n:1'][2] == 'clients'
    assert list(g.senses('customers')) == [('n:1','plural')]
