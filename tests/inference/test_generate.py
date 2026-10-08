import pytest
import torch

from src.inference import generate
from src.data.dataset import EOS_ID, PAD_ID, tokenize_pairs
from src.models.model import TextToSQLModel
from src.tokenizer import bpe


TOK = bpe.Tokenizer(bpe.load_seed_tokens())  # 병합 없이 seed + 바이트 단위


VOCAB = 300


def tiny_model(max_seq_len: int = 48) -> TextToSQLModel:
    torch.manual_seed(0)
    return TextToSQLModel(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                          max_seq_len=max_seq_len, padding_idx=PAD_ID)


@pytest.mark.unit
def test_safe_decode():
    """정상 바이트는 복원, 깨진 UTF-8 이나 vocab 밖 ID 는 None"""
    assert generate.safe_decode(TOK.encode("SELECT x"), TOK) == "SELECT x"
    assert generate.safe_decode([0xFF], TOK) is None
    assert generate.safe_decode([VOCAB + 500], TOK) is None


@pytest.mark.unit
def test_split_example_roundtrip():
    """토큰화된 예시에서 (질문, 정답 SQL) 을 그대로 복원하는지 검증"""
    pair = {"question": "what city is ashley in?", "sql": "SELECT city FROM customers WHERE name = 'ashley'"}
    ex = tokenize_pairs([pair], TOK)[0]
    assert generate.split_example(ex, TOK) == (pair["question"], pair["sql"])


def scripted_forward(script_by_first_token: dict[int, list[int]], prompt_lens: dict[int, int]):
    """forward 대체: 행의 두 번째 토큰(질문 첫 토큰)으로 행을 구분해 정해진 스크립트를 생성한다."""
    def forward(idx):
        logits = torch.zeros(idx.size(0), idx.size(1), VOCAB)
        for b in range(idx.size(0)):
            key = idx[b, 1].item()
            k = idx.size(1) - prompt_lens[key]
            script = script_by_first_token[key]
            logits[b, -1, script[k] if k < len(script) else EOS_ID] = 1.0
        return logits
    return forward


@pytest.mark.unit
def test_generate_batch_groups_by_prompt_length_and_keeps_order():
    """프롬프트 길이가 다른 예시를 섞어도 입력 순서대로, <eos> 전까지만 반환하는지 검증"""
    pairs = [{"question": "a", "sql": "x"}, {"question": "bbb", "sql": "x"}, {"question": "cc", "sql": "x"},
             {"question": "dd", "sql": "x"}]
    examples = tokenize_pairs(pairs, TOK)
    first = {ex["ids"][1]: ex["sql_start"] for ex in examples}
    scripts = {ord("a"): [65], ord("b"): [66, 66, 66], ord("c"): [67, 67], ord("d"): []}
    model = tiny_model()
    model.forward = scripted_forward(scripts, first)
    model.train()
    out = generate.generate_batch(model, examples, torch.device("cpu"), None, batch_size=1)
    assert out == [[65], [66, 66, 66], [67, 67], []]
    assert model.training, "generate_batch 후 원래 train/eval 상태로 돌아가야 함"


@pytest.mark.integration
def test_generate_batch_matches_single_generate():
    """실제 모델에서 배치 생성 결과가 예시별 model.generate 결과와 같아야 함 (오른쪽 패딩 + 길이별 묶음)"""
    pairs = [{"question": q, "sql": "x"} for q in ("hi", "hello there", "yo", "abc", "hello world")]
    examples = tokenize_pairs(pairs, TOK)
    model = tiny_model().eval()
    batch = generate.generate_batch(model, examples, torch.device("cpu"), None, max_new_tokens=8)
    for ex, got in zip(examples, batch):
        assert got == model.generate(ex["ids"][:ex["sql_start"]], eos_id=EOS_ID, max_new_tokens=8)
