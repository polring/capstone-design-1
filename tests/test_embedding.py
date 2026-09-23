
import torch
import sys
import os

# 모듈을 가져오기 위해 상위 디렉터리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN, NUM_HEADS, HEAD_DIM
from src.models.embedding import TokenEmbedding, precompute_freqs_cis, apply_rotary_emb

def test_token_embedding():
    batch_size = 4
    seq_len = 128
    
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    token_emb = TokenEmbedding(VOCAB_SIZE, HIDDEN_DIM)
    x_emb = token_emb(x)
    
    assert x_emb.shape == (batch_size, seq_len, HIDDEN_DIM), "Token Embedding 차원 오류!"

def test_rope_shape():
    batch_size = 4
    seq_len = 128
    
    xq = torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    xk = torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    
    freqs_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN)
    assert freqs_cis.shape == (SEQ_LEN, HEAD_DIM // 2), "RoPE 주파수 차원 오류!"
    
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])
    
    assert xq_rot.shape == (batch_size, seq_len, NUM_HEADS, HEAD_DIM), "RoPE 변환 후 Query 차원 오류!"
    assert xk_rot.shape == (batch_size, seq_len, NUM_HEADS, HEAD_DIM), "RoPE 변환 후 Key 차원 오류!"

def test_rope_norm_preservation():
    # 벡터 길이(Norm) 보존 검증: 회전에 의해서는 벡터의 길이가 변하지 않아야 함
    batch_size = 2
    seq_len = 16
    
    xq = torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    xk = torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    
    freqs_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN)
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])
    
    norm_xq_before = torch.linalg.norm(xq, dim=-1)
    norm_xq_after = torch.linalg.norm(xq_rot, dim=-1)
    
    assert torch.allclose(norm_xq_before, norm_xq_after, atol=1e-5), "Query 벡터 Norm이 보존되지 않습니다."
    
    norm_xk_before = torch.linalg.norm(xk, dim=-1)
    norm_xk_after = torch.linalg.norm(xk_rot, dim=-1)
    
    assert torch.allclose(norm_xk_before, norm_xk_after, atol=1e-5), "Key 벡터 Norm이 보존되지 않습니다."

def test_rope_rotation_angle():
    # 회전 각도 검증: 상대 거리가 동일한 두 토큰(벡터) 쌍의 내적(Attention Score) 값이 같아야 함
    batch_size = 1
    
    xq = torch.randn(batch_size, 1, NUM_HEADS, HEAD_DIM)
    xk = torch.randn(batch_size, 1, NUM_HEADS, HEAD_DIM)
    
    freqs_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN)
    
    # 위치 m 과 n에 대해 각각 RoPE 적용
    m = 2
    n = 5
    
    freq_m = freqs_cis[m:m+1]
    freq_n = freqs_cis[n:n+1]
    
    xq_m, _ = apply_rotary_emb(xq, xk, freq_m)
    _, xk_n = apply_rotary_emb(xq, xk, freq_n)
    
    score_mn = torch.sum(xq_m * xk_n, dim=-1)
    
    # 상대 거리가 동일한 m+k, n+k 에 대해 RoPE 적용
    k = 3
    m_k = m + k
    n_k = n + k
    
    freq_m_k = freqs_cis[m_k:m_k+1]
    freq_n_k = freqs_cis[n_k:n_k+1]
    
    xq_m_k, _ = apply_rotary_emb(xq, xk, freq_m_k)
    _, xk_n_k = apply_rotary_emb(xq, xk, freq_n_k)
    
    score_m_k_n_k = torch.sum(xq_m_k * xk_n_k, dim=-1)
    
    assert torch.allclose(score_mn, score_m_k_n_k, atol=1e-5), "상대 거리가 같은 두 쌍의 내적이 다릅니다."
