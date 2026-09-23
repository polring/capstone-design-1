import torch
import torch.nn as nn
from src.config import VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN, NUM_HEADS, HEAD_DIM

class TokenEmbedding(nn.Module):
    """
    입력 토큰 ID를 HIDDEN_DIM 차원의 실수 텐서로 매핑하는 기본 모듈
    """
    def __init__(self, vocab_size: int, hidden_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화: 정규분포 mean=0, std=0.02"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        
    def forward(self, x):
        return self.embedding(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
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
    
    # GPU(CUDA) 장치 호환성 대응을 위해 디바이스 동기화
    freqs_cis = freqs_cis.to(xq.device)
    
    # 2. 주파수 텐서의 차원 맞추기 (브로드캐스팅을 위해)
    # shape: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq_.shape[-1])
    
    # 3. 회전 변환 적용 후 다시 실수형 텐서로 복구
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # 원래 데이터 타입으로 되돌려 반환
    return xq_out.type_as(xq), xk_out.type_as(xk)
