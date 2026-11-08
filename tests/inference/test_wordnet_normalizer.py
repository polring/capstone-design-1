import pytest

from src.inference.wordnet_normalizer import WordNetNormalizer
from src.run import normalize_questions

pytestmark = pytest.mark.unit


def normalizer():
    return WordNetNormalizer(dict(format_version=1, trained_words=['items', 'customers', 'stock'],
        protected_values=['silver patrons', 'new york', 'gold'],
        mapping={'articles':['items',0], 'patrons':['customers',1], 'inventory':['stock',0]}))


def test_values_and_punctuation_are_preserved_without_labels():
    q = '''Patrons named silver patrons in new york, except 'articles' or "inventory", id >= 1500 on 2026-10-18. inventory?'''
    text, changes = normalizer().normalize(q)
    assert text == '''customers named silver patrons in new york, except 'articles' or "inventory", id >= 1500 on 2026-10-18. stock?'''
    assert [c['source'] for c in changes] == ['Patrons', 'inventory']
    assert changes[0]['similarity'] == .5
    assert all(q[c['start']:c['end']] == c['source'] for c in changes)


def test_unknown_names_in_quotes_and_contractions():
    q = "what's the stock for 'articles patrons' and patrons?"
    assert normalizer()(q) == "what's the stock for 'articles patrons' and customers?"


def test_known_words_unknown_words_and_identifier_substrings_stay_unchanged():
    q = 'items stock zzzz articles_id inventory123 patrons'
    assert normalizer()(q) == 'items stock zzzz articles_id inventory123 customers'
    assert normalizer()(normalizer()(q)) == normalizer()(q)


def test_preprocessor_interface_ignores_labels_and_does_not_mutate_records():
    pairs = [{'question':'patrons', 'sql':'never read this'}, {'question':'articles', 'sql':'different label'}]
    questions, trace = normalize_questions(pairs, normalizer())
    assert questions == ['customers','items']
    assert pairs[0]['question'] == 'patrons'
    assert trace[1]['question'] == 'articles' and trace[1]['model_question'] == 'items'


def test_map_rejects_changing_observed_words_or_unknown_targets():
    with pytest.raises(ValueError, match='observed'):
        WordNetNormalizer(dict(format_version=1, trained_words=['items'], mapping={'items':['items',0]}))
    with pytest.raises(ValueError, match='replacement'):
        WordNetNormalizer(dict(format_version=1, trained_words=['items'], mapping={'articles':['new',0]}))


def test_explicit_search_fragments_are_values_even_without_quotes():
    n = WordNetNormalizer(dict(format_version=1, metadata={'protect_like_fragments':True},
        trained_words=['items','customers'], mapping={'articles':['items',0], 'patrons':['customers',1]}))
    q = 'articles with names beginning with patrons and ending in articles; patrons whose names contain articles'
    assert n(q) == 'items with names beginning with patrons and ending in articles; customers whose names contain articles'
    assert n('articles with prefix patrons or suffix articles') == 'items with prefix patrons or suffix articles'
    assert n('articles named patrons* or %articles%') == 'items named patrons* or %articles%'


def test_terse_and_reordered_literal_cues_do_not_require_fluent_english():
    n = WordNetNormalizer(dict(format_version=1, metadata={'protect_like_fragments':True,'like_guard_version':3},
        trained_words=['items','customers'], mapping={'articles':['items',0], 'patrons':['customers',1]}))
    assert n('articles name ending patrons') == 'items name ending patrons'
    assert n('articles with names finishing in patrons') == 'items with names finishing in patrons'
    assert n('articles with patrons anywhere in their names') == 'items with patrons anywhere in their names'
    assert n('articles with patrons at the end') == 'items with patrons at the end'


def test_phrases_win_over_word_substitution_without_cascading_or_value_changes():
    from src.data.preprocessing.domain_wordnet_map import phrase_rules, symbol_rules
    n = WordNetNormalizer(dict(format_version=1, metadata={'protect_like_fragments':True,'like_guard_version':3},
        trained_words=['items','customers'], mapping={'articles':['items',3]},
        phrase_rules=phrase_rules()+symbol_rules(), protected_values=['silver cap']))
    text, changes = n.normalize("articles: record count, arithmetic mean of price, qty>=7; name 'avg desc' and silver cap")
    assert text == "items: row count, average price, quantity at least 7; name 'avg desc' and silver cap"
    assert [c['rule'] for c in changes] == ['wordnet','record_count','arithmetic_mean','qty','symbol_ge']
    assert n("stock<=10 and stock>0, price!=12.50") == 'stock at most 10 and stock more than 0, price not equal to 12.50'
    assert n('largest stock, smallest price, max 5 units') == 'highest stock, lowest price, max 5 units'


def test_phrase_rules_keep_like_values_verbatim():
    from src.data.preprocessing.domain_wordnet_map import phrase_rules
    n = WordNetNormalizer(dict(format_version=1, metadata={'protect_like_fragments':True,'like_guard_version':3},
        trained_words=[],mapping={},phrase_rules=phrase_rules()))
    assert n('names ending avg; qty 5') == 'names ending avg; quantity 5'
