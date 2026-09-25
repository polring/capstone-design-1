import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embedding import apply_rotary_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class CausalMaskedSelfAttention(nn.Module):
    """Multi-Head Causal Masked Self-Attention"""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor,freqs_cis = None) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence Length, Dim
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 1. q, k, v의 Shape를 모두 (B, T, num_heads, head_dim)으로 동일하게 정렬
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # 2. RoPE 위치 임베딩 적용
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # 3. SDPA 입력을 위해 3개 텐서 모두 동시에 (B, num_heads, T, head_dim)으로 transpose
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Attention 연산
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """FFN with SwiGLU or GELU Activation"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down
       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    


class TransformerDecoderBlock(nn.Module):
    """Pre-LN Transformer Decoder Block with RMSNorm and Causal Self-Attention"""
    def __init__(self, dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalMaskedSelfAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x: torch.Tensor,freqs_cis: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x