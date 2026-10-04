import math

import torch
import torch.nn as nn

from src.config import VOCAB_SIZE, HIDDEN_DIM, SEQ_LEN, NUM_HEADS, NUM_LAYERS, FFN_DIM
from src.models.embedding import TokenEmbedding, precompute_freqs_cis
from src.models.transformer import RMSNorm, TransformerDecoderBlock


class TextToSQLModel(nn.Module):
    """
    Decoder-only Transformer: 임베딩 → 디코더 블록 × NUM_LAYERS → RMSNorm → 출력층
    출력층은 토큰 임베딩과 가중치를 공유한다 (weight tying)
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = FFN_DIM,
        max_seq_len: int = SEQ_LEN,
        padding_idx: int | None = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = TokenEmbedding(vocab_size, dim, padding_idx=padding_idx)
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(dim)  # Pre-LN 구조라 마지막 블록 출력은 정규화되지 않은 상태
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.embedding.weight  # weight tying

        # RoPE 주파수: 학습 대상이 아니므로 state_dict 에는 넣지 않는다
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(dim // num_heads, max_seq_len), persistent=False
        )
        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int):
        """블록 내부 Linear 를 정규분포(std=0.02)로 초기화한다.
        residual 경로에 더해지는 출력 projection(out_proj, w2)은 std 를 1/sqrt(2·층수)로 줄여
        층이 쌓여도 residual 분산이 커지지 않게 한다 (GPT-2 방식).
        토큰 임베딩(= lm_head)은 TokenEmbedding 이 이미 초기화하므로 건드리지 않는다."""
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        for name, p in self.blocks.named_parameters():
            if p.dim() != 2:
                continue  # RMSNorm weight 는 1 로 둔다
            std = residual_std if name.endswith(("out_proj.weight", "w2.weight")) else 0.02
            nn.init.normal_(p, mean=0.0, std=std)

    def num_params(self) -> int:
        """학습 파라미터 수 (tying 된 임베딩/출력층은 한 번만 센다)"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) 토큰 ID → logits: (B, T, vocab_size)"""
        T = idx.size(1)
        assert T <= self.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        x = self.tok_emb(idx)
        freqs_cis = self.freqs_cis[:T]  # apply_rotary_emb 는 길이 T 짜리를 기대한다
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], eos_id: int, max_new_tokens: int = 64) -> list[int]:
        """Greedy 디코딩: prompt(<bos> question <sep>) 뒤에 이어지는 토큰을 <eos> 전까지 반환한다.
        시퀀스가 짧아(최대 64토큰 안팎) KV cache 없이 매 스텝 전체를 다시 계산한다."""
        device = self.freqs_cis.device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out: list[int] = []
        for _ in range(max_new_tokens):
            if ids.size(1) >= self.max_seq_len:
                break
            next_id = self(ids)[0, -1].argmax().item()
            if next_id == eos_id:
                break
            out.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        return out
