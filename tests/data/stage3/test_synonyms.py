import copy
import json
import random
from pathlib import Path

import pytest

from src.data.stage3 import sql_ast as a
from src.data.stage3 import compose as c
from src.data.stage3.compose import _compose_aggregate as compose, pick_template
from src.data.stage3.compose import SynonymSwapper

pytestmark = pytest.mark.integration
BASE = Path('data/stage3/composer')


def load(name):
    return json.loads((BASE / f'{name}.json').read_text(encoding='utf-8'))


def test_dictionary_checks_include_advanced_training_and_evaluation():
    data, frames, lexicon, templates = (load(name) for name in ('synonyms', 'frames', 'lexicon', 'advanced'))
    assert c.synonym_problems(data, frames, lexicon, templates) == []
    bad = copy.deepcopy(data)
    bad['advanced_words']['calculate'] = ['count']
    assert any('count' in error for error in c.synonym_problems(bad, frames, lexicon, templates))
    bad['advanced_words']['calculate'] = ['invented inactive phrase']
    bad['advanced_words']['not a training phrase'] = ['compute']
    assert any('not a training phrase' in error for error in c.synonym_problems(bad, frames, lexicon, templates))


def test_category_question_does_not_get_plural_auxiliary():
    syn = c.Synonyms({**load('synonyms'), 'prob': 1.0})
    assert syn.apply('which category does {np_def} belong to?', random.Random(0)) == 'which category does {np_def} belong to?'
    assert 'are ' in syn.apply('which order ids belong to {np_def}?', random.Random(0))


def test_context_limits_advanced_verbs_and_protects_slots():
    data = {**load('synonyms'), 'prob': 1.0}
    plain, advanced = c.Synonyms(data), c.Synonyms(data, context='advanced')
    text = 'calculate {aggregate} from {rows}'
    assert plain.apply(text, random.Random(0)) == text
    changed = advanced.apply(text, random.Random(0))
    assert changed != text and '{aggregate}' in changed and '{rows}' in changed
    assert advanced.apply('recalculate {aggregate}', random.Random(0)) == 'recalculate {aggregate}'


@pytest.mark.parametrize('text', [
    'for a report on {rows}, group by {group}',
    'i need a list of {rows}',
    'find out the {aggregate} for {rows}',
    'carry out the calculation for {rows}',
    'the list price is {value}',
])
def test_nouns_and_phrasal_verbs_are_not_rewritten_as_request_verbs(text):
    syn = c.Synonyms({**load('synonyms'), 'prob': 1.0}, context='advanced')
    changed = syn.apply(text, random.Random(0))
    if 'a report' in text:
        assert 'a report' in changed
    elif 'a list' in text:
        assert 'a list' in changed
    elif 'find out' in text:
        assert 'find out' in changed
    elif 'carry out' in text:
        assert 'carry out' in changed
    else:
        assert 'list price' in changed


def test_advanced_eval_template_is_not_augmented():
    data = c.Synonyms({'prob': 1.0, 'words': {'return': ['INVALID']}})
    templates = {'aggregate': {'train': ['return {aggregate}'], 'eval': ['return {aggregate}']}}
    assert pick_template(templates, 'aggregate', 'eval', random.Random(0), [], data) == 'return {aggregate}'
    assert pick_template(templates, 'aggregate', 'train', random.Random(0), [], data) == 'INVALID {aggregate}'


@pytest.mark.parametrize('sql', [
    "SELECT SUM(quantity) FROM orders WHERE customer_id = 1452 AND order_date >= '2026-01-01'",
    "SELECT category, AVG(price) FROM items WHERE stock > 0 GROUP BY category HAVING AVG(price) >= 20 ORDER BY AVG(price) DESC, category ASC LIMIT 3",
    "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items WHERE item_name = 'wireless mouse')",
])
def test_advanced_augmentation_preserves_filters_values_and_reference(sql):
    data = {**load('synonyms'), 'prob': 1.0}
    composer = c.Composer(load('frames'), load('lexicon'), 'train', data)
    record = a.to_dict(a.parse(sql))
    questions = [compose(record, random.Random(seed), composer)[0] for seed in range(30)]
    assert len(set(questions)) > 1
    for question in questions:
        assert '{' not in question
        if '1452' in sql:
            assert '1452' in question and '2026-01-01' in question
        if 'wireless mouse' in sql:
            assert 'wireless mouse' in question
        if 'HAVING' in sql:
            assert '20' in question and '3' in question and 'at least' in question
        assert record['sql'] == sql


def test_epoch_augmentation_protects_literal_that_is_itself_a_synonym():
    sql = "SELECT * FROM items WHERE item_name = 'catalog'"
    pair = {'question': "pull up all items named catalog", 'sql': sql, 'sql_group': sql, 'source': 'llm:gpt'}
    changed, stats = SynonymSwapper().swap([pair], 1, random.Random(0))
    assert changed[0]['synonym_swapped']
    assert changed[0]['question'].endswith('named catalog')
    assert changed[0]['sql'] == sql and changed[0]['sql_group'] == sql
    assert stats['swapped'] == 1


def test_epoch_augmentation_handles_advanced_inner_values_and_leaves_composer_pairs_alone():
    sql = "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items WHERE item_name = 'catalog')"
    pairs = [{'question': "return item names with price above the mean price of catalog", 'sql': sql,
              'source': source} for source in ('llm:gpt', 'composer_advanced')]
    changed, stats = SynonymSwapper().swap(pairs, 1, random.Random(0))
    assert changed[0]['synonym_swapped'] and changed[0]['question'].endswith('of catalog')
    assert changed[1]['question'] == pairs[1]['question'] and not changed[1]['synonym_swapped']
    assert all(p['sql'] == sql for p in changed)
    assert stats['llm_pairs'] == 1


def test_epoch_augmentation_preserves_numbers_and_is_deterministic():
    pair = {'question': "calculate the total ordered quantity for customer 1452 before 2026-01-01",
            'sql': "SELECT SUM(quantity) FROM orders WHERE customer_id = 1452 AND order_date < '2026-01-01'",
            'source': 'llm:gpt'}
    swapper = SynonymSwapper()
    first = swapper.swap([pair], 1, random.Random(0))
    assert first == swapper.swap([pair], 1, random.Random(0))
    assert '1452 before 2026-01-01' in first[0][0]['question']
    assert first[0][0]['sql'] == pair['sql']
    assert swapper.swap([pair], 0, random.Random(0))[0][0]['question'] == pair['question']


def test_epoch_augmentation_swaps_composer_pairs_with_composer_prob():
    sql = "SELECT item_name FROM items WHERE price > (SELECT AVG(price) FROM items WHERE item_name = 'catalog')"
    pairs = [{'question': "return item names with price above the mean price of catalog", 'sql': sql,
              'source': source} for source in ('composer_advanced', 'llm:gpt')]
    changed, stats = SynonymSwapper().swap(pairs, 0, random.Random(0), composer_prob=1)
    assert changed[0]['synonym_swapped'] and changed[0]['question'].endswith('of catalog')
    assert not changed[0]['question'].startswith('return ')
    assert changed[1]['question'] == pairs[1]['question']
    assert all(p['sql'] == sql for p in changed)
    assert stats['composer_swapped'] == 1
