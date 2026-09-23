from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

VOCAB_SIZE = 1024
BYTE_BASE = 0
SPECIAL_BASE = 256
SEED_BASE = 260

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]

SEED_TOKENS = ["WHERE", "FROM", "SELECT",
                "customers", "customer_id", "name", "city", "membership",
                "items", "item_id", "item_name", "category", "price", "stock",
                "orders", "order_id", "quantity", "status",
                "new york", "chicago", "houston", "seattle", "boston", "denver",
                "bronze", "silver", "gold", "platinum",
                "electronics", "office supplies", "kitchen", "furniture", "clothing", "sports",
                "pending", "shipped", "delivered", "cancelled"]

MERGE_BASE = SEED_BASE + len(SEED_TOKENS)

def _validate_seeds(seeds: list[str]) -> None:
    assert len(seeds) == len(set(seeds)), "Seed tokens must be unique."
    for s in seeds:
        assert len(s) >= 2, f"Seed token '{s}' must be at least 2 characters long."
        assert s.strip() == s, f"Seed token '{s}' must not have leading or trailing whitespace."

_validate_seeds(SEED_TOKENS)

SEED_TO_ID: dict[str, int] = {s: SEED_BASE + i for i, s in enumerate(SEED_TOKENS)}
ID_TO_SEED: dict[int, str] = {i: s for s, i in SEED_TO_ID.items()}

MERGE_BUDGET = VOCAB_SIZE - MERGE_BASE

Piece = tuple[str, "int | None"]

def _build_pattern(seeds: list[str]) -> re.Pattern:
    alts: list[str] = []
    for s in sorted(seeds, key=len, reverse=True):
        alts.append(r"\b" + re.escape(s) + r"\b")      # 단어 경계로 둘러쌓여 있는 시드 토큰 (' . , * 등으로 둘러쌓여 있는 경우)
    alts.append(r"[^\W\d_]+")                          # 단어문자이면서 숫자와 밑줄이 아닌것, 즉 알파벳만
    alts.append(r"[0-9]")                              # 0~9 까지 한글자만
    alts.append(r"\s")                                 # 공백 문자
    alts.append(r".")                                  # 아무 문자 한글자
    return re.compile("|".join(alts), re.DOTALL)       # re.DOTALL: .이 줄바꿈 문자를 포함한 모든 문자와 매치되도록 함

_PATTERN = None

def pre_tokenize(text: str) -> list[Piece]:
    global _PATTERN
    if _PATTERN is None:
        _PATTERN = _build_pattern(SEED_TOKENS)
    pieces: list[Piece] = []
    pos = 0
    for m in _PATTERN.finditer(text):
        assert m.start() == pos, f"Unexpected gap in text at position {pos}."
        s = m.group(0)
        pieces.append((s, SEED_TO_ID.get(s)))
        pos = m.end()
    assert pos == len(text), f"Unexpected gap in text at position {pos}."
    return pieces

def pieces_to_text(pieces: list[Piece]) -> str:
    return "".join(p[0] for p in pieces)

def check_roundtrip(text: str) -> None:
    pieces = pre_tokenize(text)
    restored = pieces_to_text(pieces)
    if restored != text:
        raise AssertionError(f"Roundtrip failed: '{restored}' != '{text}'")

def show(text: str) -> None:
    pieces = pre_tokenize(text)
    check_roundtrip(text)
    parts = []
    for s, sid in pieces:
        parts.append(f"[{s}]" if sid is None else repr(s))
    print(f"Text: {text}\\n")
    print(f"Pieces: {' '.join(parts)}\\n")
    print(f"조각 {len(pieces)}개, Seed 토큰 {sum(1 for _, sid in pieces if sid is not None)}개\\n")

def load_corpus(data_dir: str | Path = "data") -> list[str]:
    corpus: list[str] = []
    for pf in sorted(Path(data_dir).glob("pilot*/pilot_train_pairs.json")):
        with open(pf, encoding="utf-8") as f:
            pairs = json.load(f)
        for row in pairs:
            corpus.append(row["question"])
            corpus.append(row["sql"])
    return corpus

def corpus_piece_freqs(corpus: list[str]) -> Counter[str]:
    freqs: Counter[str] = Counter()
    for text in corpus:
        for piece, sid in pre_tokenize(text):
            if sid is None:
                freqs[piece] += 1
    return freqs

def get_pair_counts(word_syms: dict[str, list[int]], freqs: Counter[str]) -> Counter[tuple[int, int]]:
    pair_counts: Counter[tuple[int, int]] = Counter()
    for word, syms in word_syms.items():
        weight = freqs[word]
        for a, b in zip(syms, syms[1:]):
            pair_counts[(a, b)] += weight
    return pair_counts

def merge_word(syms: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i = 0
    while i < len(syms):
        if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(syms[i])
            i += 1
    return out

def build_base_id_to_bytes() -> dict[int, bytes]:
    id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for sid, s in ID_TO_SEED.items():
        id_to_bytes[sid] = s.encode("utf-8")
    return id_to_bytes

Merge = tuple[tuple[int, int], int]

def train(corpus: list[str], merge_budget: int = MERGE_BUDGET) -> tuple[list[Merge], dict[int, bytes]]:
    freqs = corpus_piece_freqs(corpus)
    word_syms: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in freqs}
    id_to_bytes = build_base_id_to_bytes()
    merges: list[Merge] = []

    for step in range(merge_budget):
        pair_counts = get_pair_counts(word_syms, freqs)
        if not pair_counts:
            break
        best_count = max(pair_counts.values())
        best_pair = min(p for p, c in pair_counts.items() if c == best_count)  # 동점이면 항상 같은 쌍이 뽑히도록 사전순 최솟값으로 고정
        new_id = MERGE_BASE + step
        for w in word_syms:
            word_syms[w] = merge_word(word_syms[w], best_pair, new_id)
        id_to_bytes[new_id] = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
        merges.append((best_pair, new_id))

    return merges, id_to_bytes

def id_to_bytes_from_merges(merges: list[Merge]) -> dict[int, bytes]:
    id_to_bytes = build_base_id_to_bytes()
    for (a, b), new_id in merges:
        id_to_bytes[new_id] = id_to_bytes[a] + id_to_bytes[b]
    return id_to_bytes

def apply_merges(syms: list[int], merges: list[Merge]) -> list[int]:
    for pair, new_id in merges:
        syms = merge_word(syms, pair, new_id)
    return syms

def encode(text: str, merges: list[Merge]) -> list[int]:
    ids: list[int] = []
    for piece, sid in pre_tokenize(text):
        if sid is not None:
            ids.append(sid)
        else:
            ids.extend(apply_merges(list(piece.encode("utf-8")), merges))
    return ids

def decode(ids: list[int], id_to_bytes: dict[int, bytes]) -> str:
    parts: list[str] = []
    buf = bytearray()
    for i in ids:
        if SPECIAL_BASE <= i < SPECIAL_BASE + len(SPECIAL_TOKENS):
            if buf:
                parts.append(bytes(buf).decode("utf-8"))
                buf = bytearray()
            parts.append(SPECIAL_TOKENS[i - SPECIAL_BASE])
        else:
            buf.extend(id_to_bytes[i])
    if buf:
        parts.append(bytes(buf).decode("utf-8"))
    return "".join(parts)

def save_merges(merges: list[Merge], path: str | Path) -> None:
    data = [[a, b, new_id] for (a, b), new_id in merges]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_merges(path: str | Path) -> list[Merge]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [((a, b), new_id) for a, b, new_id in data]

if __name__ == "__main__":
    corpus = load_corpus()
    print(f"코퍼스 크기: {len(corpus)}")

    merges, id_to_bytes = train(corpus)
    print(f"학습된 병합 수: {len(merges)} / {MERGE_BUDGET}")
    save_merges(merges, "bpe_merges.json")

    bad = 0
    for text in corpus:
        restored = decode(encode(text, merges), id_to_bytes)
        if restored != text:
            bad += 1
            if bad <= 5:
                print(f"ROUNDTRIP FAIL: {text!r} -> {restored!r}")
    print(f"round-trip 실패: {bad} / {len(corpus)}")

    holdout_path = Path("data/holdout.json")
    if holdout_path.exists():
        with open(holdout_path, encoding="utf-8") as f:
            holdout = json.load(f)
        names = [r["name"] for r in holdout["customers"]["heldout_rows"]]
        names += [r["item_name"] for r in holdout["items"]["heldout_rows"]]
        counts = [len(encode(n, merges)) for n in names]
        print(f"미등장 이름 {len(names)}개 평균 토큰 수: {sum(counts) / len(counts):.2f}")

