import sys
import os

# 프로젝트 루트 디렉터리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import bpe


# pre_tokenize가 SEED_TOKENS를 일반 알파벳 조각으로 쪼개지 않고 하나의 시드 토큰으로 인식하는지 검증
def test_seed_token_priority():
    # 시드 토큰은 단어 경계 안에서 하나의 조각으로 인식돼야 함
    pieces = bpe.pre_tokenize("SELECT name FROM customers")
    tokens = [s for s, _ in pieces]
    assert "SELECT" in tokens
    assert "customers" in tokens


# pre_tokenize/pieces_to_text가 빈 문자열·공백·한글 혼용 텍스트에서도 원문을 그대로 복원하는지 검증
def test_roundtrip_various_text():
    samples = [
        "SELECT * FROM orders WHERE status = 'shipped'",
        "고객 이름: Alice, 도시: new york",
        "",
        "   ",
    ]
    for text in samples:
        bpe.check_roundtrip(text)  # 내부에서 다르면 AssertionError 발생


# train이 동일 입력에 대해 항상 같은 병합 순서를 만드는지 검증 (동점 쌍을 사전순 최솟값으로 고정하는 로직 확인)
def test_train_merges_deterministic():
    corpus = ["SELECT name FROM customers", "SELECT name FROM customers"]
    merges1, _ = bpe.train(corpus, merge_budget=5)
    merges2, _ = bpe.train(corpus, merge_budget=5)
    assert merges1 == merges2, "동일 입력에 대해 병합 결과가 달라지면 안 됨 (동점 처리가 결정적이어야 함)"


# 코퍼스로 학습한 merges로 encode한 뒤 decode하면 원문 텍스트가 그대로 복원되는지 검증
def test_encode_decode_roundtrip():
    corpus = ["SELECT name FROM customers WHERE city = 'seattle'"]
    merges, id_to_bytes = bpe.train(corpus, merge_budget=10)

    text = "SELECT name FROM customers WHERE city = 'seattle'"
    ids = bpe.encode(text, merges)
    restored = bpe.decode(ids, id_to_bytes)

    assert restored == text


# decode가 <bos>/<eos> 같은 특수 토큰 id를 바이트로 잘못 디코딩하지 않고 태그 문자열로 복원하는지 검증
def test_special_tokens_decode_correctly():
    id_to_bytes = bpe.build_base_id_to_bytes()
    bos = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<bos>")
    eos = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<eos>")

    ids = [bos] + list(b"hi") + [eos]
    decoded = bpe.decode(ids, id_to_bytes)

    assert decoded == "<bos>hi<eos>"


# merge_word가 한 시퀀스 안에서 같은 쌍이 여러 번 등장할 때 전부 병합하는지 검증
def test_merge_word_replaces_all_occurrences():
    # 'ab ab' 패턴이 여러 번 등장해도 모두 병합되어야 함
    syms = [1, 2, 3, 2, 3]
    merged = bpe.merge_word(syms, (2, 3), 99)
    assert merged == [1, 99, 99]


# merge_word가 병합 후 인덱스를 건너뛰어서 겹치는 쌍을 재사용하지 않는지 검증
def test_merge_word_scans_left_to_right_without_overlap():
    # 병합 후 위치를 건너뛰므로 겹치는 쌍은 재사용되지 않음 (1,1,1) -> merge(0,1) -> [99, 1]
    syms = [1, 1, 1]
    merged = bpe.merge_word(syms, (1, 1), 99)
    assert merged == [99, 1]


# merge_word가 대상 쌍이 없을 때 입력을 그대로 반환하는지 검증
def test_merge_word_no_match_returns_unchanged():
    syms = [1, 2, 3]
    merged = bpe.merge_word(syms, (5, 6), 99)
    assert merged == syms


# merge_word가 빈 리스트/원소 1개짜리 리스트 같은 경계 입력에서 에러 없이 그대로 반환하는지 검증
def test_merge_word_empty_and_single_element():
    assert bpe.merge_word([], (1, 2), 99) == []
    assert bpe.merge_word([1], (1, 2), 99) == [1]


# save_merges로 저장했다가 load_merges로 다시 읽었을 때 tuple 구조까지 원본과 동일하게 복원되는지 검증
def test_save_load_merges_roundtrip(tmp_path):
    merges = [((1, 2), 300), ((300, 3), 301), ((10, 20), 302)]
    path = tmp_path / "merges.json"

    bpe.save_merges(merges, path)
    loaded = bpe.load_merges(path)

    assert loaded == merges
