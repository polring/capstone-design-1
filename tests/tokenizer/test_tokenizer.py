import json
import sqlite3

import pytest

from src import config
from src.tokenizer import bpe

SEEDS = bpe.load_seed_tokens()
TOK = bpe.Tokenizer(SEEDS)  # 병합 없이 seed + 바이트 단위


# pre_tokenize가 seed 토큰을 일반 알파벳 조각으로 쪼개지 않고 하나의 시드 토큰으로 인식하는지 검증
@pytest.mark.unit
def test_seed_token_priority():
    # 시드 토큰은 단어 경계 안에서 하나의 조각으로 인식돼야 함
    pieces = TOK.pre_tokenize("SELECT name FROM customers")
    tokens = [s for s, _ in pieces]
    assert "SELECT" in tokens
    assert "customers" in tokens
    assert dict(pieces)["SELECT"] == TOK.seed_to_id["SELECT"]


# pre_tokenize가 빈 문자열·공백·한글 혼용 텍스트에서도 원문을 그대로 복원하는지 검증
@pytest.mark.unit
def test_roundtrip_various_text():
    samples = [
        "SELECT * FROM orders WHERE status = 'shipped'",
        "고객 이름: Alice, 도시: new york",
        "",
        "   ",
    ]
    for text in samples:
        assert "".join(p for p, _ in TOK.pre_tokenize(text)) == text
        assert TOK.decode(TOK.encode(text)) == text


# train이 동일 입력에 대해 항상 같은 병합 순서를 만드는지 검증 (동점 쌍을 사전순 최솟값으로 고정하는 로직 확인)
@pytest.mark.unit
def test_train_merges_deterministic():
    corpus = ["SELECT name FROM customers", "SELECT name FROM customers"]
    vocab = TOK.merge_base + 5
    tok1 = bpe.Tokenizer.train(corpus, SEEDS, vocab_size=vocab)
    tok2 = bpe.Tokenizer.train(corpus, SEEDS, vocab_size=vocab)
    assert tok1.merges == tok2.merges, "동일 입력에 대해 병합 결과가 달라지면 안 됨 (동점 처리가 결정적이어야 함)"


# 코퍼스로 학습한 토크나이저로 encode한 뒤 decode하면 원문 텍스트가 그대로 복원되는지 검증
@pytest.mark.unit
def test_encode_decode_roundtrip():
    text = "which city does ashley live in? SELECT city FROM customers WHERE name = 'ashley'"
    tok = bpe.Tokenizer.train([text], SEEDS, vocab_size=TOK.merge_base + 10)
    assert len(tok.merges) > 0
    assert tok.decode(tok.encode(text)) == text


# decode가 <bos>/<eos> 같은 특수 토큰 id를 바이트로 잘못 디코딩하지 않고 태그 문자열로 복원하는지 검증
@pytest.mark.unit
def test_special_tokens_decode_correctly():
    bos = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<bos>")
    eos = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<eos>")
    assert TOK.decode([bos] + list(b"hi") + [eos]) == "<bos>hi<eos>"


# merge_word가 한 시퀀스 안에서 같은 쌍이 여러 번 등장할 때 전부 병합하는지 검증
@pytest.mark.unit
def test_merge_word_replaces_all_occurrences():
    # 'ab ab' 패턴이 여러 번 등장해도 모두 병합되어야 함
    syms = [1, 2, 3, 2, 3]
    merged = bpe.merge_word(syms, (2, 3), 99)
    assert merged == [1, 99, 99]


# merge_word가 병합 후 인덱스를 건너뛰어서 겹치는 쌍을 재사용하지 않는지 검증
@pytest.mark.unit
def test_merge_word_scans_left_to_right_without_overlap():
    # 병합 후 위치를 건너뛰므로 겹치는 쌍은 재사용되지 않음 (1,1,1) -> merge(0,1) -> [99, 1]
    syms = [1, 1, 1]
    merged = bpe.merge_word(syms, (1, 1), 99)
    assert merged == [99, 1]


# merge_word가 대상 쌍이 없을 때 입력을 그대로 반환하는지 검증
@pytest.mark.unit
def test_merge_word_no_match_returns_unchanged():
    syms = [1, 2, 3]
    merged = bpe.merge_word(syms, (5, 6), 99)
    assert merged == syms


# merge_word가 빈 리스트/원소 1개짜리 리스트 같은 경계 입력에서 에러 없이 그대로 반환하는지 검증
@pytest.mark.unit
def test_merge_word_empty_and_single_element():
    assert bpe.merge_word([], (1, 2), 99) == []
    assert bpe.merge_word([1], (1, 2), 99) == [1]


# 저장했다가 다시 읽은 토크나이저가 같은 seed·병합 규칙과 같은 인코딩을 내는지 검증
@pytest.mark.unit
def test_save_load_roundtrip(tmp_path):
    text = "which city does ashley live in? SELECT city FROM customers WHERE name = 'ashley'"
    tok = bpe.Tokenizer.train([text], SEEDS, vocab_size=TOK.merge_base + 10)
    tok.save(tmp_path / "tokenizer.json")
    loaded = bpe.Tokenizer.load(tmp_path / "tokenizer.json")
    assert loaded.seed_tokens == tok.seed_tokens and loaded.merges == tok.merges
    assert loaded.encode(text) == tok.encode(text)


# 예전 형식(병합 규칙만 있는 리스트)은 seed 파일의 목록과 함께 읽는지 검증
@pytest.mark.unit
def test_load_legacy_merges_list(tmp_path):
    merges = [[104, 105, TOK.merge_base]]
    (tmp_path / "bpe_merges.json").write_text(json.dumps(merges), encoding="utf-8")
    tok = bpe.Tokenizer.load(tmp_path / "bpe_merges.json")
    assert tok.seed_tokens == SEEDS
    assert tok.encode("hi") == [TOK.merge_base]


# 저장 파일의 특수 토큰 배치가 코드와 다르면 거부하는지 검증
@pytest.mark.unit
def test_load_rejects_other_special_tokens(tmp_path):
    d = TOK.to_dict()
    d["special_tokens"] = ["<pad>", "<bos>", "<eos>"]
    (tmp_path / "t.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        bpe.Tokenizer.load(tmp_path / "t.json")


# 커밋된 토크나이저가 seed 파일과 같은 seed 목록으로 학습됐는지 검증 (seed 파일만 고치고 다시 학습하지 않은 경우)
@pytest.mark.unit
def test_committed_tokenizer_matches_seed_file():
    tok = bpe.Tokenizer.load(config.TOKENIZER_PATH)
    assert tok.seed_tokens == SEEDS
    assert tok.vocab_size <= config.VOCAB_SIZE


# seed 토큰이 실제 DB 의 테이블·컬럼 이름과 범주형 값을 모두 포함하는지 검증 (db_gen 과 seed 파일이 어긋나지 않게)
@pytest.mark.unit
def test_seed_tokens_cover_db_schema_and_categories():
    con = sqlite3.connect(config.DB_PATH)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    expected = set(tables)
    for t in tables:
        expected |= {r[1] for r in con.execute(f"PRAGMA table_info({t})")}
    for t, c in [("customers", "city"), ("customers", "membership"), ("items", "category"), ("orders", "status")]:
        expected |= {r[0] for r in con.execute(f"SELECT DISTINCT {c} FROM {t}")}
    con.close()
    assert expected <= set(SEEDS), sorted(expected - set(SEEDS))
