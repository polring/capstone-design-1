import json
from pathlib import Path
from collections import Counter

import pytest

pytestmark=pytest.mark.integration
BASE=Path('data/stage3/composer')
ROLES=('aggregate','group','group_default','subquery','subquery_default')


def load(name):return json.loads((BASE/name).read_text(encoding='utf-8'))


def frame_entries(value):
    if isinstance(value,list):
        for row in value:yield from frame_entries(row)
    elif isinstance(value,dict):
        if 'text' in value:yield value
        else:
            for key,v in value.items():
                if not key.startswith('_'):yield from frame_entries(v)


def test_sentence_budget_and_train_eval_ratio_exclude_suffixes():
    frames=list(frame_entries(load('frames.json')))
    counts=Counter(row.get('split','train') for row in frames)
    advanced=load('advanced.json')
    for role in ROLES:
        for split in ('train','eval'):counts[split]+=len(advanced[role][split])
    assert counts=={'train':528,'eval':72}
    assert sum(counts.values())==600


def test_each_template_has_aligned_style_and_training_only_source_provenance():
    provenance_path = Path('data_raw/stage3/composer/template_provenance.json')
    if not provenance_path.is_file():
        pytest.skip('Local template provenance is not included in the repository')
    library=load('advanced.json');provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    assert len(provenance['sentences'])==291
    seen=set()
    for p in provenance['sentences']:
        role,split,index=p['role'],p['split'],p['index']
        assert library[role][split][index]==p['text']
        assert library['template_meta'][role][split][index]['style']==p['style']
        assert p['style'] in ('common','conversational','contextual')
        assert p['text'] not in seen
        seen.add(p['text'])
        if split=='eval':assert p['source_pair_id']==0 and p['source_question'] is None
        elif p['source_pair_id']:
            assert p['source_question'] and p['source_sql']
    assert provenance['pilot_derived_train']>0
