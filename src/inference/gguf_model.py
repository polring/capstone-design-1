# 역할: 팀의 디코더 블록으로 GGUF 내보내기용 모델과 체크포인트 계약을 구현한다.
# 사용처: exporter.py에서 학습 가중치를 C 추론기용 모델 구조로 읽어 변환할 때 사용한다.

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


def _has_training_checkpoint_format(checkpoint):
    return isinstance(checkpoint, dict) and {"model_cfg", "model"} <= checkpoint.keys()


# TextToSQLModel uses the same blocks; only these top-level names differ.
_TRAINING_NAME_PREFIXES = (("tok_emb.", "embedding."), ("norm.", "final_norm."))


def _rename_training_parameter(name):
    for old, new in _TRAINING_NAME_PREFIXES:
        if name.startswith(old):
            return new + name[len(old) :]
    return name


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


def from_training_checkpoint(checkpoint):
    """Convert a TextToSQLModel checkpoint (training best.pt or release model.pt).

    eps and theta keep their defaults, which match RMSNorm and RoPE in the team model.
    Unused vocabulary rows are removed so the existing GGUF contract exposes
    exactly the training model's output candidates.
    """
    cfg = checkpoint["model_cfg"]
    vocab_size = cfg.get("valid_vocab_size", cfg["vocab_size"])
    if vocab_size is None:
        vocab_size = cfg["vocab_size"]
    if type(vocab_size) is not int or not 0 < vocab_size <= cfg["vocab_size"]:
        raise ValueError("valid_vocab_size must be within the allocated vocabulary")
    config = ModelConfig(
        vocab_size=vocab_size,
        dim=cfg["dim"],
        layers=cfg["num_layers"],
        heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        context=cfg["max_seq_len"],
    )
    state = {
        _rename_training_parameter(name): tensor
        for name, tensor in checkpoint["model"].items()
    }
    for name in ("embedding.embedding.weight", "lm_head.weight"):
        if state[name].shape != (cfg["vocab_size"], cfg["dim"]):
            raise ValueError(f"wrong training shape: {name}")
        state[name] = state[name][:vocab_size]
    model = InferenceModel(config)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if _has_training_checkpoint_format(checkpoint):
        return from_training_checkpoint(checkpoint)
    if not _has_checkpoint_contract(checkpoint):
        raise ValueError(
            "checkpoint requires config and state_dict, or model_cfg and model;"
            " see docs/exporter-loader.md"
        )
    model = InferenceModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()
