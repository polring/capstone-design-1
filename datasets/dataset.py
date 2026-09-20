from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

import bpe_tokenizer as bpe

BOS_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<bos>")
EOS_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<eos>")
SEP_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<sep>")
PAD_ID = bpe.SPECIAL_BASE + bpe.SPECIAL_TOKENS.index("<pad>")

Pair = dict[str, str]
Example = dict[str, "list[int] | int"]

def load_pairs(data_dir: str | Path, glob: str) -> list[Pair]:
    pairs: list[Pair] = []
    for pf in sorted(Path(data_dir).glob(glob)):
        with open(pf, encoding="utf-8") as f:
            pairs.extend(json.load(f))
    return pairs

def load_train_pairs(data_dir: str | Path = "db/data") -> list[Pair]:
    return load_pairs(data_dir, "pilot*/pilot_train_pairs.json")

def load_eval_indist_pairs(data_dir: str | Path = "db/data") -> list[Pair]:
    return load_pairs(data_dir, "eval_indist*/pilot_train_pairs.json")

def encode_pair(question: str, sql: str, merges: list[bpe.Merge]) -> list[int]:
    q_ids = bpe.encode(question, merges)
    s_ids = bpe.encode(sql, merges)
    return [BOS_ID] + q_ids + [SEP_ID] + s_ids + [EOS_ID]

def tokenize_pairs(pairs: list[Pair], merges: list[bpe.Merge]) -> list[Example]:
    examples: list[Example] = []
    for p in pairs:
        ids = encode_pair(p["question"], p["sql"], merges)
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

if __name__ == "__main__":
    merges = bpe.load_merges("bpe_merges.json")

    train_pairs = load_train_pairs()
    eval_pairs = load_eval_indist_pairs()
    print(f"train pairs: {len(train_pairs)}, eval_indist pairs: {len(eval_pairs)}")

    train_examples = tokenize_pairs(train_pairs, merges)
    eval_examples = tokenize_pairs(eval_pairs, merges)

    lengths = [len(e["ids"]) for e in train_examples + eval_examples]
    lengths.sort()
    n = len(lengths)
    print(f"sequence length (<bos> question <sep> sql <eos>): "
          f"min={lengths[0]} max={lengths[-1]} mean={sum(lengths) / n:.1f} "
          f"p50={lengths[n // 2]} p99={lengths[int(n * 0.99)]}")

    context_len = 128
    over = sum(1 for L in lengths if L > context_len)
    print(f"sequences exceeding context length {context_len}: {over} / {n} ({over / n:.2%})")

    id_to_bytes = bpe.id_to_bytes_from_merges(merges)
    bad = 0
    for p, ex in zip(train_pairs[:5] + eval_pairs[:5], train_examples[:5] + eval_examples[:5]):
        ids = ex["ids"]
        restored_q = bpe.decode(ids[1:ids.index(SEP_ID)], id_to_bytes)
        restored_s = bpe.decode(ids[ex["sql_start"]:-1], id_to_bytes)
        if restored_q != p["question"] or restored_s != p["sql"]:
            bad += 1
            print(f"MISMATCH: {p!r} -> q={restored_q!r} s={restored_s!r}")
        else:
            print(f"ids[:12]={ids[:12]}... len={len(ids)}  ok: {p['question']!r} | {p['sql']!r}")
    print(f"sample round-trip failures: {bad} / 10")

    print()
    sample = train_examples[:8]
    batch = collate_fn(sample)
    print(f"collate_fn batch shapes: input_ids={tuple(batch['input_ids'].shape)} "
          f"target_ids={tuple(batch['target_ids'].shape)} loss_mask={tuple(batch['loss_mask'].shape)}")

    ok = all(
        batch["loss_mask"][i].sum().item() == len(sample[i]["ids"]) - sample[i]["sql_start"]
        for i in range(len(sample))
    )
    print(f"loss_mask true-count matches expected SQL+<eos> length for every row: {ok}")

    from torch.utils.data import DataLoader
    loader = DataLoader(TextToSQLDataset(train_examples), batch_size=32, shuffle=True, collate_fn=collate_fn)
    real_batch = next(iter(loader))
    print(f"DataLoader batch (bs=32, shuffled): input_ids={tuple(real_batch['input_ids'].shape)} "
          f"(max length in this random batch, vs. fixed context {context_len})")
