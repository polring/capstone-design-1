import torch
import torch.nn as nn
from config import VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN, NUM_HEADS, HEAD_DIM

class TokenEmbedding(nn.Module):
    """
    입력 토큰 ID를 HIDDEN_DIM 차원의 실수 텐서로 매핑하는 기본 모듈
    """
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
    def forward(self, x):
        return self.embedding(x)

def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    RoPE를 위한 주파수(frequency) 미리 계산
    dim: HEAD_DIM
    end: SEQ_LEN
    """
    # 회전 주파수(Frequency) 계산 로직
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    
    # 극좌표계를 이용해 복소수 형태로 변환 (cos + i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Query와 Key 텐서에 RoPE 회전 변환을 수행하는 함수
    xq, xk shape: (batch_size, seq_len, num_heads, head_dim)
    freqs_cis shape: (seq_len, head_dim // 2)
    """
    # 1. 실수형 텐서를 복소수형으로 변환
    # (인접한 두 차원을 복소수의 실수부와 허수부로 취급)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. 주파수 텐서의 차원 맞추기 (브로드캐스팅을 위해)
    # shape: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq_.shape[-1])
    
    # 3. 회전 변환 적용 후 다시 실수형 텐서로 복구
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # 원래 데이터 타입으로 되돌려 반환
    return xq_out.type_as(xq), xk_out.type_as(xk)

if __name__ == "__main__":
    # --- 단위 테스트 (Unit Test) ---
    print("=== Configuration ===")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"SEQ_LEN: {SEQ_LEN}")
    print(f"NUM_HEADS: {NUM_HEADS}")
    print(f"HEAD_DIM: {HEAD_DIM}")
    print("=====================\n")

    batch_size = 4
    seq_len = 128 # 가변적인 테스트를 위해 128로 설정 (최대 SEQ_LEN 이하)

    # 1. Dummy Input (Token IDs 생성 - torch.randint 활용)
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    print(f"[Input] Token IDs shape: {x.shape}")

    # 2. Token Embedding 검증
    token_emb = TokenEmbedding(VOCAB_SIZE, HIDDEN_DIM)
    x_emb = token_emb(x)
    print(f"[Token Embedding] Output shape: {x_emb.shape} (Expected: {batch_size}, {seq_len}, {HIDDEN_DIM})")
    assert x_emb.shape == (batch_size, seq_len, HIDDEN_DIM), "Token Embedding 차원 오류!"

    # 3. Dummy Query & Key for RoPE
    # Attention 모듈 내부에서 분할된 차원으로 가정: (batch_size, seq_len, num_heads, head_dim)
    xq = torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    xk = torch.randn(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    print(f"\n[Query/Key before RoPE] Shape: {xq.shape}")

    # 4. RoPE 주파수 생성 및 형태 검증
    freqs_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN)
    print(f"[RoPE] precompute_freqs_cis shape: {freqs_cis.shape} (Expected: {SEQ_LEN}, {HEAD_DIM // 2})")
    assert freqs_cis.shape == (SEQ_LEN, HEAD_DIM // 2), "RoPE 주파수 차원 오류!"

    # 5. RoPE 적용 검증
    # 텐서 시퀀스 길이에 맞게 freqs_cis 슬라이싱 적용 (seq_len 만큼)
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis[:seq_len])
    print(f"[Query after RoPE] Shape: {xq_rot.shape} (Expected: {batch_size}, {seq_len}, {NUM_HEADS}, {HEAD_DIM})")
    print(f"[Key after RoPE] Shape: {xk_rot.shape} (Expected: {batch_size}, {seq_len}, {NUM_HEADS}, {HEAD_DIM})")
    assert xq_rot.shape == (batch_size, seq_len, NUM_HEADS, HEAD_DIM), "RoPE 변환 후 Query 차원 오류!"
    assert xk_rot.shape == (batch_size, seq_len, NUM_HEADS, HEAD_DIM), "RoPE 변환 후 Key 차원 오류!"
    
    print("\n✅ 모든 단위 테스트 통과 (Dimension Mismatch 없음)!")
