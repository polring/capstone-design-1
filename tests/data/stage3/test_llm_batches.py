import json
from pathlib import Path

import pytest

from src.data.stage3.llm_batches import normalized, validate_batch
from src.data.stage3 import sql_ast

pytestmark=pytest.mark.unit


def test_code_normalizes_new_sql_without_mutating_original_label():
    sql="SELECT order_id FROM orders WHERE quantity > 3 ORDER BY order_date DESC, order_id DESC LIMIT 3"
    record=sql_ast.to_dict(sql_ast.parse(sql))
    updated=normalized(record)
    assert record['sql']==sql
    assert updated['sql'].endswith('ORDER BY order_date DESC, order_id ASC LIMIT 3')


def fixture_batch(tmp_path):
    source=[{'sql':f'SELECT stock FROM items WHERE item_id = {3000+i}'} for i in range(25)]
    rows=[{'id':i+1,'sql':r['sql'],'questions':[f'question {i} variant {j}' for j in range(5)]} for i,r in enumerate(source)]
    (tmp_path/'round_sql.json').write_text(json.dumps(source),encoding='utf-8')
    (tmp_path/'responses').mkdir()
    path=tmp_path/'responses/batch_01.json'
    path.write_text(json.dumps({'records':rows}),encoding='utf-8')
    return rows,path


def test_batch_rejects_sql_relabeling_even_when_all_counts_match(tmp_path):
    rows,path=fixture_batch(tmp_path)
    validate_batch(tmp_path,1)
    rows[0]['sql']=rows[0]['sql'].replace('stock','price')
    path.write_text(json.dumps({'records':rows}),encoding='utf-8')
    with pytest.raises(AssertionError,match='SQL changed'):validate_batch(tmp_path,1)


def test_batch_rejects_duplicate_ids_and_missing_questions(tmp_path):
    rows,path=fixture_batch(tmp_path)
    rows[1]['id']=rows[0]['id']
    path.write_text(json.dumps({'records':rows}),encoding='utf-8')
    with pytest.raises(AssertionError,match='IDs'):validate_batch(tmp_path,1)
    rows[1]['id']=2
    rows[0]['questions']=rows[0]['questions'][:4]
    path.write_text(json.dumps({'records':rows}),encoding='utf-8')
    with pytest.raises(AssertionError,match='five'):validate_batch(tmp_path,1)


def test_custom_batch_size_question_count_and_partial_last_batch(tmp_path):
    source = [{'sql': f'SELECT stock FROM items WHERE item_id = {3000+i}'} for i in range(3)]
    (tmp_path/'round_sql.json').write_text(json.dumps(source), encoding='utf-8')
    (tmp_path/'round_meta.json').write_text(json.dumps({'sql_count': 3, 'batch_size': 2, 'questions_per_sql': 2}), encoding='utf-8')
    (tmp_path/'responses').mkdir()
    for batch, ids in ((1, (1, 2)), (2, (3,))):
        rows = [{'id': i, 'sql': source[i-1]['sql'], 'questions': [f'question {i} a', f'question {i} b']} for i in ids]
        (tmp_path/'responses'/f'batch_{batch:02}.json').write_text(json.dumps({'records': rows}), encoding='utf-8')
        validate_batch(tmp_path,batch)


@pytest.mark.parametrize("question, sql, status, exit_code", [
    ("what is the maximum stock of items?", "SELECT MAX(stock) FROM items", "pass", 0),
    ("what is the average stock of items?", "SELECT MAX(stock) FROM items", "violation", 1),
    ("show items with stock in the vicinity of 50", "SELECT * FROM items WHERE stock > 50", "review", 1),
])
def test_audit_separates_review_and_violation_without_promoting(monkeypatch, question, sql, status, exit_code):
    """세 판정을 별도 파일로 보고하고 미검토 후보의 자동 편입을 막는다."""
    from types import SimpleNamespace
    from src.data.stage3 import llm_batches as m
    from src.data import db_gen
    from src.tokenizer.bpe import Tokenizer

    row = sql_ast.to_dict(sql_ast.parse(sql))
    pair = dict(question=question, sql=sql, sql_id=1, family="test", style=0)
    saved = {}
    monkeypatch.setattr(m, "round_config", lambda _: (1, 1, 1, 1))
    monkeypatch.setattr(m, "read", lambda p: [pair] if str(p).endswith("pairs_candidate.json") else
                        [row] if str(p).endswith("round_sql.json") else [])
    monkeypatch.setattr(m, "save", lambda p, value: saved.__setitem__(Path(p).name, value))
    monkeypatch.setattr(db_gen, "load_bans", lambda _: {"items": {}})
    monkeypatch.setattr(m.sqlite3, "connect", lambda *a, **kw: SimpleNamespace(
        execute=lambda _: SimpleNamespace(fetchall=lambda: []), close=lambda: None))
    monkeypatch.setattr(Tokenizer, "load", lambda _: SimpleNamespace(encode=lambda text: text.split()))
    args = SimpleNamespace(out="unused", pairs=None, holdout="holdout", eval_sql="evaluation", exclude=[],
                           db="database", tokenizer="tokenizer")
    assert m.audit(args) == exit_code
    assert saved["quality_report.json"]["promoted_to_training"] is False
    assert len(saved["review_flags.json"]) == (status == "review")
    assert len(saved["quality_flags.json"]) == (status == "violation")
