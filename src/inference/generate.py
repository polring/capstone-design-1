"""
generate.py — 질문 → SQL 생성 (greedy)

프롬프트 <bos> 질문 <sep> 뒤를 배치로 생성하고 토큰 ID 를 SQL 문자열로 되돌린다. 평가(evaluation/scoring.py),
배치 추론(run.py), 시연(demo.py) 이 같은 생성 로직을 쓴다. 제약 디코더는 없다.
"""

from __future__ import annotations

import torch

from src.data.dataset import BOS_ID, EOS_ID, PAD_ID, SEP_ID
from src.tokenizer import bpe


@torch.no_grad()
def generate_batch(model, examples: list[dict], device, amp_dtype,
                   batch_size: int = 512, max_new_tokens: int = 64) -> list[list[int]]:
    """각 예시의 프롬프트(<bos> question <sep>) 뒤를 greedy 로 생성해 SQL 토큰 ID 를 반환한다 (<eos> 제외).

    오른쪽 패딩 + causal attention 구조라 길이가 다른 프롬프트를 한 배치에 넣을 수 없다
    (왼쪽 패딩은 attention mask 가 없어 패딩을 보게 된다). 그래서 프롬프트 길이(= sql_start)가
    같은 것끼리 묶어 배치 생성한다. 반환 순서는 입력 순서와 같다.
    """
    was_training = model.training
    model.eval()
    groups: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex["sql_start"], []).append(i)

    outputs: list[list[int]] = [[] for _ in examples]
    for plen, idxs in groups.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            ids = torch.tensor([examples[i]["ids"][:plen] for i in chunk], dtype=torch.long, device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            for _ in range(min(max_new_tokens, model.max_seq_len - plen)):
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    next_id = model(ids)[:, -1].argmax(-1)
                next_id = torch.where(done, torch.full_like(next_id, PAD_ID), next_id)
                ids = torch.cat([ids, next_id[:, None]], dim=1)
                done |= next_id == EOS_ID
                if done.all():
                    break
            for i, row in zip(chunk, ids[:, plen:].tolist()):
                gen = row[:row.index(EOS_ID)] if EOS_ID in row else row
                outputs[i] = [t for t in gen if t != PAD_ID]
    model.train(was_training)
    return outputs


def predict_questions(model, questions: list[str], tok: bpe.Tokenizer, device, amp_dtype,
                      batch_size: int = 512) -> list[str | None]:
    """정답 없는 질문 목록을 배치로 SQL 로 바꾼다. 질문은 학습 데이터처럼 소문자로 바꿔 넣는다.
    프롬프트가 모델 최대 길이를 넘거나 디코딩할 수 없는 출력이면 None."""
    examples, idx = [], []
    for i, q in enumerate(questions):
        ids = [BOS_ID] + tok.encode(q.strip().lower()) + [SEP_ID]
        if len(ids) < model.max_seq_len:
            examples.append({"ids": ids, "sql_start": len(ids)})
            idx.append(i)
    preds: list[str | None] = [None] * len(questions)
    for i, out in zip(idx, generate_batch(model, examples, device, amp_dtype, batch_size)):
        preds[i] = safe_decode(out, tok)
    return preds


def safe_decode(ids: list[int], tok: bpe.Tokenizer) -> str | None:
    """생성 결과가 깨진 UTF-8 이거나 토크나이저에 없는 ID 면 None (항상 오답 처리).
    특수 토큰은 "<sep>" 같은 문자열로 복원되므로 None 은 아니지만 정답 SQL 과 일치할 수 없어 역시 오답이 된다."""
    try:
        return tok.decode(ids)
    except (UnicodeDecodeError, KeyError):
        return None


def split_example(ex: dict, tok: bpe.Tokenizer) -> tuple[str, str]:
    """토큰화된 예시에서 (질문, 정답 SQL) 문자열을 복원한다."""
    plen = ex["sql_start"]
    return tok.decode(ex["ids"][1:plen - 1]), tok.decode(ex["ids"][plen:-1])
