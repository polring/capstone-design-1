"""Checkpoint contract built using the team's decoder blocks; no pretrained weights."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from src.models.embedding import TokenEmbedding, precompute_freqs_cis
from src.models.transformer import RMSNorm, TransformerDecoderBlock


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1024
    dim: int = 256
    layers: int = 6
    heads: int = 4
    ffn_dim: int = 768
    context: int = 128
    eps: float = 1e-6
    theta: float = 10000.0

    def __post_init__(self):
        if not self._has_positive_integer_dimensions():
            raise ValueError("dimensions must be positive integers")
        if not self._has_compatible_attention_dimensions():
            raise ValueError(
                "vocab >= 256; dim divisible by heads; even head dimension required"
            )
        if not self._has_valid_numeric_parameters():
            raise ValueError("eps and theta must be finite and positive")

    def _has_positive_integer_dimensions(self):
        return all(
            type(value) is int and value > 0
            for value in (
                self.vocab_size,
                self.dim,
                self.layers,
                self.heads,
                self.ffn_dim,
                self.context,
            )
        )

    def _has_compatible_attention_dimensions(self):
        """Called after dimensions are validated, so the head count is positive."""
        return (
            self.vocab_size >= 256
            and self.dim % self.heads == 0
            and (self.dim // self.heads) % 2 == 0
        )

    def _has_valid_numeric_parameters(self):
        return all(
            math.isfinite(value) and value > 0 for value in (self.eps, self.theta)
        )


def _is_input_shape_within_context(ids, context):
    return ids.ndim == 2 and 0 < ids.shape[1] <= context


def _has_checkpoint_contract(checkpoint):
    return (
        isinstance(checkpoint, dict) and {"config", "state_dict"} <= checkpoint.keys()
    )


class InferenceModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(config.dim, config.heads, config.ffn_dim)
                for _ in range(config.layers)
            ]
        )
        for block in self.blocks:
            block.attn_norm.eps = block.ffn_norm.eps = config.eps
        self.final_norm = RMSNorm(config.dim, config.eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.register_buffer(
            "freqs",
            precompute_freqs_cis(
                config.dim // config.heads, config.context, config.theta
            ),
            persistent=False,
        )

    def forward(self, ids):
        if not _is_input_shape_within_context(ids, self.config.context):
            raise ValueError("expected [batch, sequence] within context")
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, self.freqs[: ids.shape[1]])
        return self.lm_head(self.final_norm(x))

    def checkpoint(self):
        return {"config": asdict(self.config), "state_dict": self.state_dict()}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not _has_checkpoint_contract(checkpoint):
        raise ValueError(
            "checkpoint requires config and state_dict; see docs/exporter-loader.md"
        )
    model = InferenceModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()
