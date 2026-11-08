"""Stage 3 uses the same public train/tokenize/run pipeline as earlier stages."""
import json

import pytest
import torch

from src import config, run, train
from src.tokenizer import bpe
from src.inference.loading import load_model
from src.evaluation.scoring import load_rules

pytestmark = pytest.mark.integration


def test_common_stage3_training_and_inference(tmp_path):
    pairs = [
        {'question': 'number of orders', 'sql': 'SELECT COUNT(*) FROM orders'},
        {'question': 'average stock', 'sql': 'SELECT AVG(stock) FROM items'},
    ]
    data = tmp_path / 'pairs.json'
    data.write_text(json.dumps(pairs), encoding='utf-8')
    tokenizer = tmp_path / 'tokenizer.json'
    tok = bpe.Tokenizer(bpe.load_seed_tokens(config.stage_paths(3).seed_tokens_path))
    tok.save(tokenizer)
    output = tmp_path / 'training'
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert train.main(['--stage', '3', '--train-files', str(data), '--tokenizer', str(tokenizer),
                           '--out', str(output), '--cpu', '--overfit', '2', '--epochs', '1',
                           '--layers', '1', '--dim', '16', '--heads', '2', '--ffn-dim', '32',
                           '--seq-len', '256', '--model-vocab-size', '512', '--batch-size', '2']) == 0
        model, ckpt = load_model(output / 'best.pt', torch.device('cpu'))
        assert ckpt['stage'] == 3
        assert model.max_seq_len == 256 and model.valid_vocab_size == tok.vocab_size
        assert run.main(['--stage', '3', '--model', str(output), '--cpu', '--no-exec',
                         '--input', 'number of orders']) == 0
        assert load_rules(3).ORDERED_EXECUTION
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('step_cap', [None, 2])
def test_round_repeat_controls_real_training_and_checkpoint(tmp_path, monkeypatch, step_cap):
    """공통 학습에서 numbers만 1회·다른 LLM 3회를 적용하고 설정을 체크포인트에 보존한다"""
    pairs = [
        {'question': 'number of orders', 'sql': 'SELECT COUNT(*) FROM orders', 'source': 'composer'},
        {'question': 'two categories', 'sql': 'SELECT category, COUNT(*) FROM items GROUP BY category ORDER BY COUNT(*) DESC, category ASC LIMIT 2', 'source': 'llm:gpt', 'round': 'numbers'},
        {'question': 'average stock', 'sql': 'SELECT AVG(stock) FROM items', 'source': 'llm:gpt', 'round': 'intent'},
    ]
    data, val, tokenizer, output = (tmp_path / name for name in ('train.json', 'val.json', 'tokenizer.json', 'output'))
    data.write_text(json.dumps(pairs), encoding='utf-8')
    val.write_text(json.dumps([{'question': 'sum stock', 'sql': 'SELECT SUM(stock) FROM items'}]), encoding='utf-8')
    bpe.Tokenizer(bpe.load_seed_tokens(config.stage_paths(3).seed_tokens_path)).save(tokenizer)
    monkeypatch.setattr(train, 'list_sets', lambda directory: [])
    options = ['--stage', '3', '--train-files', str(data), '--val-files', str(val),
               '--tokenizer', str(tokenizer), '--out', str(output), '--cpu', '--epochs', '1',
               '--layers', '1', '--dim', '16', '--heads', '2', '--ffn-dim', '32',
               '--seq-len', '256', '--model-vocab-size', '512', '--batch-size', '1',
               '--name-swap-ratio', '0', '--llm-synonym-prob', '0',
               '--llm-repeat', '3', '--llm-round-repeat', 'numbers=1']
    if step_cap is not None:
        options += ['--steps-per-epoch', str(step_cap)]
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert train.main(options) == 0
        ckpt = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        assert ckpt['step'] == (step_cap or 5)  # composer 1 + numbers 1 + intent 3, optionally capped
        assert ckpt['args']['llm_round_repeat'] == {'numbers': 1}
        changed = [v if v != 'numbers=1' else 'numbers=2' for v in options]
        with pytest.raises(SystemExit, match='LLM 반복 설정'):
            train.main(changed + ['--resume', str(output / 'last.pt')])
    finally:
        torch.set_num_threads(old_threads)


def test_common_tokenizer_accepts_stage_and_explicit_paths(tmp_path):
    data = tmp_path / 'pairs.json'
    data.write_text(json.dumps([{'question': 'number of orders', 'sql': 'SELECT COUNT(*) FROM orders'}]), encoding='utf-8')
    output = tmp_path / 'tokens.json'
    assert bpe.main(['--stage', '3', '--train-files', str(data), '--out', str(output),
                     '--vocab-size', '512', '--holdout', str(tmp_path / 'absent.json')]) == 0
    tok = bpe.Tokenizer.load(output)
    assert 'COUNT' in tok.seed_tokens
    assert json.loads(output.with_name('training_report.json').read_text(encoding='utf-8'))['roundtrip_errors'] == 0


def test_llm_prepare_custom_counts_without_external_calls(tmp_path):
    from src.data.stage3.llm_batches import main

    out = tmp_path / "round"
    # 생성 안내문은 로컬 작업 파일이므로 테스트에서 직접 준비한다.
    brief = tmp_path / "brief.txt"
    brief.write_text(
        "Author {questions_per_sql} independently worded questions per SQL.\n"
        "Use one session for up to {batch_size} SQL.\n",
        encoding="utf-8",
    )
    main(["prepare", "--out", str(out), "--n-sql", "3", "--batch-size", "2",
          "--questions-per-sql", "2", "--seed", "0", "--brief", str(brief)])
    meta = json.loads((out / "round_meta.json").read_text(encoding="utf-8"))
    assert meta["expected_pairs"] == 6
    prompts = sorted((out / "prompts").glob("*.txt"))
    assert len(prompts) == 2
    for prompt in prompts:
        text = prompt.read_text(encoding="utf-8")
        assert "Author 2 independently" in text
        assert "up to 2 SQL" in text
        assert "five distinct questions" not in text
    assert len(json.loads(prompts[-1].read_text(encoding="utf-8").split("INPUT:\n")[1])) == 1


def test_default_composer_build_keeps_frozen_training_data_untouched(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.data.stage3 import compose, sql_ast

    sql_dir, train_dir, raw_dir = tmp_path / "sql", tmp_path / "train", tmp_path / "raw"
    sql_dir.mkdir(parents=True)
    train_dir.mkdir()
    (sql_dir / "train.json").write_text(json.dumps([
        sql_ast.to_dict(sql_ast.parse(sql)) for sql in
        ("SELECT * FROM items WHERE stock > 0", "SELECT COUNT(*) FROM orders")
    ]), encoding="utf-8")
    frozen = train_dir / "pairs.json"
    frozen.write_text('[{"question":"original","sql":"SELECT * FROM items"}]', encoding="utf-8")
    original = frozen.read_bytes()
    monkeypatch.setattr(compose, "PATHS", SimpleNamespace(sql_dir=sql_dir, train_dir=train_dir, raw_dir=raw_dir))
    assert compose.main(["build", "--epochs", "1"]) == 0
    pairs = json.loads((raw_dir / "composer/pairs.json").read_text(encoding="utf-8"))
    assert len(pairs) == 2
    assert frozen.read_bytes() == original

    explicit = tmp_path / "custom.json"
    assert compose.main(["build", "--epochs", "1", "--sql", str(sql_dir / "train.json"), "--out", str(explicit)]) == 0
    assert len(json.loads(explicit.read_text(encoding="utf-8"))) == 2
    assert frozen.read_bytes() == original

