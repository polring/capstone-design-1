import math

import pytest
import torch

from src.data.dataset import EOS_ID, PAD_ID
from src.models.model import TextToSQLModel

# ---------------------------------------------------------------------------
# Test Configuration & Fixtures
# ---------------------------------------------------------------------------
VOCAB, DIM, LAYERS, HEADS, FFN, MAX_LEN = 300, 64, 2, 4, 160, 32  # 특수 토큰(256~259)을 포함하는 최소 vocab


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = TextToSQLModel(vocab_size=VOCAB, dim=DIM, num_layers=LAYERS, num_heads=HEADS,
                       ffn_dim=FFN, max_seq_len=MAX_LEN, padding_idx=PAD_ID)
    return m.eval()


def scripted_forward(prompt_len: int, script: list[int]):
    """forward 대체: 생성 위치 k 에서 script[k] 를 argmax 로 내는 logits 를 돌려준다 (script 가 끝나면 EOS)."""
    def forward(idx):
        k = idx.size(1) - prompt_len
        nxt = script[k] if k < len(script) else EOS_ID
        logits = torch.zeros(idx.size(0), idx.size(1), VOCAB)
        logits[:, -1, nxt] = 1.0
        return logits
    return forward


# ===========================================================================
# Unit Tests
# ===========================================================================

@pytest.mark.unit
def test_forward_shape(model):
    """forward 출력이 (B, T, vocab) 규격인지 검증"""
    idx = torch.randint(0, 256, (3, 10))
    assert model(idx).shape == (3, 10, VOCAB)


@pytest.mark.unit
def test_weight_tying(model):
    """출력층과 토큰 임베딩이 같은 텐서를 공유하는지 검증"""
    assert model.lm_head.weight is model.tok_emb.embedding.weight


@pytest.mark.unit
def test_num_params_counts_tied_weight_once(model):
    """num_params 가 tying 된 임베딩/출력층을 한 번만 세는지 검증"""
    unique = {id(p): p.numel() for _, p in model.named_parameters(remove_duplicate=False)}
    assert model.num_params() == sum(unique.values())
    assert model.num_params() < sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))


@pytest.mark.unit
def test_rope_buffer_not_in_state_dict(model):
    """RoPE 주파수(freqs_cis)는 학습 대상이 아니므로 state_dict 에 저장되지 않아야 함"""
    assert "freqs_cis" not in model.state_dict()


@pytest.mark.unit
def test_residual_projection_init_std():
    """out_proj / w2 는 std 0.02/sqrt(2L), 나머지 블록 행렬은 std 0.02 로 초기화되는지 검증"""
    torch.manual_seed(0)
    m = TextToSQLModel(vocab_size=VOCAB, dim=256, num_layers=6, num_heads=4, ffn_dim=683, max_seq_len=MAX_LEN)
    residual_std = 0.02 / math.sqrt(2 * 6)
    for name, p in m.blocks.named_parameters():
        if p.dim() != 2:
            assert torch.all(p == 1.0), f"{name}: RMSNorm weight 는 1 이어야 함"
            continue
        expected = residual_std if name.endswith(("out_proj.weight", "w2.weight")) else 0.02
        assert p.std().item() == pytest.approx(expected, rel=0.1), name


@pytest.mark.unit
def test_causality(model):
    """t 위치 토큰을 바꿔도 t 이전 위치의 logits 는 그대로여야 함 (causal)"""
    idx = torch.randint(0, 256, (1, 12))
    changed = idx.clone()
    changed[0, 8] = (changed[0, 8] + 1) % 256
    with torch.no_grad():
        a, b = model(idx), model(changed)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6)
    assert not torch.allclose(a[:, 8:], b[:, 8:])


@pytest.mark.unit
def test_right_padding_does_not_change_real_positions(model):
    """오른쪽 패딩은 attention mask 없이도 실제 토큰 위치의 logits 를 바꾸지 않아야 함"""
    idx = torch.randint(0, 256, (1, 10))
    padded = torch.cat([idx, torch.full((1, 5), PAD_ID)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(idx), model(padded)[:, :10], atol=1e-6)


@pytest.mark.unit
def test_forward_rejects_too_long_sequence(model):
    """max_seq_len 을 넘는 입력은 AssertionError"""
    with pytest.raises(AssertionError):
        model(torch.zeros(1, MAX_LEN + 1, dtype=torch.long))


@pytest.mark.unit
def test_generate_stops_at_eos(model):
    """generate 가 <eos> 직전까지의 토큰만 반환하는지 검증 (<eos> 미포함)"""
    prompt = [257, 10, 11, 259]
    model.forward = scripted_forward(len(prompt), [65, 66, 67])
    assert model.generate(prompt, eos_id=EOS_ID) == [65, 66, 67]


@pytest.mark.unit
def test_generate_stops_at_max_seq_len(model):
    """<eos> 가 나오지 않으면 max_seq_len 에서 멈추는지 검증"""
    prompt = [257, 10, 259]
    model.forward = scripted_forward(len(prompt), [65] * 100)
    assert len(model.generate(prompt, eos_id=EOS_ID)) == MAX_LEN - len(prompt)


@pytest.mark.unit
def test_generate_respects_max_new_tokens(model):
    """max_new_tokens 상한을 지키는지 검증"""
    prompt = [257, 10, 259]
    model.forward = scripted_forward(len(prompt), [65] * 100)
    assert model.generate(prompt, eos_id=EOS_ID, max_new_tokens=5) == [65] * 5


# ===========================================================================
# Integration Tests
# ===========================================================================

@pytest.mark.integration
def test_backward_reaches_all_parameters(model):
    """loss.backward() 후 모든 파라미터에 gradient 가 흐르는지 검증"""
    model.train()
    idx = torch.randint(0, 256, (2, 12))
    logits = model(idx)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), idx.flatten()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
