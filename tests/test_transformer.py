import pytest
import torch

from src.models.transformer import (
    RMSNorm,
    CausalMaskedSelfAttention,
    FeedForward,
    TransformerDecoderBlock,
)

# ---------------------------------------------------------------------------
# Test Configuration & Fixtures
# ---------------------------------------------------------------------------
B, T, C = 2, 8, 64  # Batch, Sequence Length, Dim
NUM_HEADS = 4
HIDDEN_DIM = 256


# ===========================================================================
# Unit Tests (단일 모듈 동작, 수학적 특성, Edge Case 검증)
# ===========================================================================

@pytest.mark.unit
def test_rmsnorm_shape_and_scale():
    """RMSNorm 출력 규격 및 값 유효성 검증"""
    norm = RMSNorm(C)
    x = torch.randn(B, T, C) * 5.0
    out = norm(x)

    assert out.shape == (B, T, C), "RMSNorm output shape mismatch"
    assert not torch.isnan(out).any(), "RMSNorm returned NaN values"


@pytest.mark.unit
def test_rmsnorm_mathematical_property():
    """RMSNorm 정규화(RMS ≒ 1.0) 수학적 올바름 검증"""
    norm = RMSNorm(C, eps=1e-6)
    x = torch.randn(B, T, C) * 10.0
    out = norm(x)

    # 마지막 차원(C) 기준 RMS 계산: sqrt(mean(x^2))
    rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
    torch.testing.assert_close(
        rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4,
        msg="RMSNorm output does not properly normalize to unit RMS"
    )


@pytest.mark.unit
def test_rmsnorm_zero_input_edge_case():
    """입력이 모두 0일 때 Epsilon에 의해 Division by Zero가 방지되는지 검증"""
    norm = RMSNorm(C)
    x_zero = torch.zeros(B, T, C)
    out = norm(x_zero)

    assert not torch.isnan(out).any(), "RMSNorm returned NaN for all-zero input"


@pytest.mark.unit
def test_causal_masked_self_attention():
    """Self-Attention 출력 규격 및 Strict Causality(미래 토큰 참조 불가) 검증"""
    attn = CausalMaskedSelfAttention(dim=C, num_heads=NUM_HEADS)
    x = torch.randn(B, T, C)
    
    out = attn(x)
    assert out.shape == (B, T, C), "Self-Attention output shape mismatch"

    # Causality Check: 미래 토큰(t > 2)을 수정해도 과거 출력(t <= 2)에 영향을 주지 않아야 함
    x_mod = x.clone()
    x_mod[:, 3:, :] += 10.0
    
    out_orig = attn(x)
    out_mod = attn(x_mod)

    torch.testing.assert_close(
        out_orig[:, :3, :], 
        out_mod[:, :3, :], 
        msg="Causal masking violated: future tokens leaked into the past"
    )


@pytest.mark.unit
def test_feed_forward():
    """FFN(SwiGL) 출력 규격 검증"""
    ffn = FeedForward(dim=C, hidden_dim=HIDDEN_DIM)
    x = torch.randn(B, T, C)
    out = ffn(x)

    assert out.shape == (B, T, C), f"FFN output shape mismatch"


@pytest.mark.unit
@pytest.mark.parametrize("seq_len", [1, 16, 128])
def test_variable_sequence_length(seq_len):
    """다양한 입력 시퀀스 길이에 대한 가변 대응성 검증"""
    block = TransformerDecoderBlock(dim=C, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM)
    x = torch.randn(B, seq_len, C)
    out = block(x)

    assert out.shape == (B, seq_len, C), f"Failed to handle sequence length {seq_len}"


# ===========================================================================
# Integration Tests (모듈 간 결합, Forward/Backward 파이프라인 검증)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.parametrize("use_swiglu", [True, False])
def test_transformer_decoder_block_forward_and_backward(use_swiglu):
    """End-to-End Forward 및 Backward (Gradient전파) decoder block 전체 검증"""
    block = TransformerDecoderBlock(
        dim=C, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM
    )
    x = torch.randn(B, T, C, requires_grad=True)

    # 1. Forward Pass
    out = block(x)
    assert out.shape == (B, T, C), "Decoder Block output shape mismatch"

    # 2. Loss & Backward Pass
    loss = out.sum()
    loss.backward()

    # 3. Input Gradient Verification
    assert x.grad is not None, "Gradients were not propagated back to inputs"
    assert not torch.isnan(x.grad).any(), "NaN detected in input gradients"

    # 4. Internal Model Parameters Gradient Verification
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of parameter {name}"