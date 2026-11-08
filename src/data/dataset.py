# 역할: 질문·SQL 쌍 로딩, 토큰화, 배치 패딩과 SQL 구간 손실 마스크를 구현한다.
# 사용처: 학습 데이터 준비와 모델 학습·검증용 배치를 구성할 때 사용한다.

from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.config import TRAIN_DIR, EVAL_DIR
from src.data.json_io import json_files, read_json
from src.tokenizer import bpe

BOS_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<bos>")
EOS_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<eos>")
SEP_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<sep>")
PAD_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<pad>")

Pair = dict[str, str]
Example = dict[str, "list[int] | int"]

def load_pairs(data_dir: str | Path, glob: str) -> list[Pair]:
    pairs: list[Pair] = []
    for pf in sorted(Path(data_dir).glob(glob)):
        pairs.extend(read_json(pf))
    return pairs

def train_files(files: list[str | Path] | None = None, train_dir: str | Path = TRAIN_DIR) -> list[Path]:
    """명시한 입력 또는 학습 폴더의 JSON·gzip JSON (BPE와 같은 규칙)."""
    return [Path(f) for f in files] if files else json_files(train_dir)

def load_train_pairs(files: list[str | Path] | None = None, train_dir: str | Path = TRAIN_DIR) -> list[Pair]:
    """train_files() 의 파일들을 순서대로 합친다."""
    pairs: list[Pair] = []
    for pf in train_files(files, train_dir):
        pairs.extend(read_json(pf))
    return pairs

def load_eval_pairs(name: str, eval_dir: str | Path = EVAL_DIR) -> list[Pair]:
    """평가셋 name 의 쌍 (폴더 방식 <eval_dir>/<name>/pairs.json, 합친 방식 <eval_dir>/pairs.json 의 set — eval_sets)"""
    from src.evaluation.eval_sets import load_eval_set
    return load_eval_set(name, eval_dir)[0]

def encode_pair(question: str, sql: str, tok: bpe.Tokenizer) -> list[int]:
    q_ids = tok.encode(question)
    s_ids = tok.encode(sql)
    return [BOS_ID] + q_ids + [SEP_ID] + s_ids + [EOS_ID]

def tokenize_pairs(pairs: list[Pair], tok: bpe.Tokenizer) -> list[Example]:
    examples: list[Example] = []
    for p in pairs:
        ids = encode_pair(p["question"], p["sql"], tok)
        examples.append({"ids": ids, "sql_start": ids.index(SEP_ID) + 1})
    return examples

class TextToSQLDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

def collate_fn(batch: list[Example]) -> dict[str, torch.Tensor]:
    """Right-pads a batch to its own max length, then builds the standard
    next-token-prediction shift (input = ids[:-1], target = ids[1:]).
    loss_mask is True only at target positions that are a real (non-pad) SQL
    token or the final <eos> — i.e. loss is computed on the SQL side only,
    never on <bos>/question/<sep> or on padding."""
    max_len = max(len(ex["ids"]) for ex in batch)
    B = len(batch)

    padded = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    loss_mask_full = torch.zeros((B, max_len), dtype=torch.bool)
    for i, ex in enumerate(batch):
        ids, sql_start, real_len = ex["ids"], ex["sql_start"], len(ex["ids"])
        padded[i, :real_len] = torch.tensor(ids, dtype=torch.long)
        loss_mask_full[i, sql_start:real_len] = True  # SQL tokens + final <eos>

    input_ids = padded[:, :-1]
    target_ids = padded[:, 1:]
    loss_mask = loss_mask_full[:, 1:]  # shift to align with target_ids
    return {"input_ids": input_ids, "target_ids": target_ids, "loss_mask": loss_mask}
