"""Serializable layout of the team's byte-level BPE vocabulary.

This describes token IDs, not a replacement tokenizer or a public-model adapter.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from src.tokenizer import bpe


@dataclass(frozen=True)
class TokenizerConfig:
    seeds: tuple[str, ...]
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    seed_base: int
    merge_base: int

    @classmethod
    def from_team_bpe(cls):
        special_ids = {
            token: bpe.SPECIAL_BASE + index
            for index, token in enumerate(bpe.SPECIAL_TOKENS)
        }
        return cls(
            seeds=tuple(bpe.SEED_TOKENS),
            pad_id=special_ids["<pad>"],
            bos_id=special_ids["<bos>"],
            eos_id=special_ids["<eos>"],
            sep_id=special_ids["<sep>"],
            seed_base=bpe.SEED_BASE,
            merge_base=bpe.SEED_BASE + len(bpe.SEED_TOKENS),
        )

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["seeds"] = tuple(values["seeds"])
        return cls(**values)

    def validate(self, vocab_size):
        scalar_ids = (
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.sep_id,
            self.seed_base,
            self.merge_base,
        )
        if any(type(value) is not int or value < 256 for value in scalar_ids):
            raise ValueError("tokenizer IDs must be integers outside byte IDs 0..255")
        if not self.seed_base <= self.merge_base <= vocab_size:
            raise ValueError("tokenizer ranges exceed vocabulary")
        if len(self.seeds) > self.merge_base - self.seed_base:
            raise ValueError("seed range overlaps merge range")
        if any(
            not isinstance(seed, str)
            or len(seed.encode("utf-8")) < 2
            or not seed.isascii()
            or seed.strip() != seed
            or "\0" in seed
            for seed in self.seeds
        ):
            raise ValueError(
                "seeds must be nonempty ASCII strings without edge whitespace"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seed")
        special_ids = scalar_ids[:4]
        if len(set(special_ids)) != 4:
            raise ValueError("special token IDs must be distinct")
        if any(
            value >= self.merge_base
            or self.seed_base <= value < self.seed_base + len(self.seeds)
            for value in special_ids
        ):
            raise ValueError("special token IDs overlap tokenizer ranges")

    def known_token_ids(self):
        return set(range(256)) | set(
            range(self.seed_base, self.seed_base + len(self.seeds))
        )
