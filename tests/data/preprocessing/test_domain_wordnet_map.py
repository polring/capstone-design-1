from collections import Counter, defaultdict

import pytest

from src.data.preprocessing import domain_wordnet_map as dm
from src.data.preprocessing.wordnet_map import WordNetGraph

pytestmark = pytest.mark.unit


def test_domain_targets_are_nouns_commercial_sense_is_required_and_ties_abstain(monkeypatch):
    anchors = {
        'items':dict(singular='item',plural='items',nodes=['n:1'],max_distance=2),
        'customers':dict(singular='customer',plural='customers',nodes=['n:2'],max_distance=2),
        'orders':dict(singular='order',plural='orders',nodes=['n:3'],max_distance=2),
    }
    monkeypatch.setattr(dm,'ANCHORS',anchors)
    g = WordNetGraph.__new__(WordNetGraph)
    g.words = {p:defaultdict(set) for p in ['n','v','a','r']}
    g.words['n'].update({'item':{'n:1'},'customer':{'n:2'},'order':{'n:3'},
                         'supply':{'n:4'},'record':{'n:5'},'exchange':{'n:6'},'subject':{'n:7'}})
    g.exceptions = {p:{} for p in g.words}
    g.words['n']['exchanges'] = {'n:8'}  # separate lexical sense of plural spelling
    g.exceptions['n']['exchanges'] = ['exchange']
    g.edges = defaultdict(set,{'n:1':{'n:4','n:7'},'n:2':{'n:7'},'n:3':{'n:5','n:6'},
                               'n:4':{'n:1'},'n:5':{'n:3'},'n:6':{'n:3'},'n:7':{'n:1','n:2'}})
    glosses = {'n:'+str(i):'' for i in range(1,9)}
    glosses.update({'n:5':'a legal statement','n:6':'a commercial exchange of goods'})
    counts = Counter(item=10,items=20,customer=10,customers=20,order=10,orders=20)
    mapping, _ = dm.domain_mapping(g,counts,glosses)
    assert mapping['supply'] == ['item',1]
    assert mapping['supplies'] == ['items',1]
    assert mapping['exchange'] == ['order',1]
    assert mapping['exchanges'] == ['orders',1]
    assert 'record' not in mapping
    assert 'subject' not in mapping
    assert not set(mapping) & set(counts)
