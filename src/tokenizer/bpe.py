"""
bpe.py — byte-level BPE 토크나이저

토큰 ID 배치 (모든 단계 공통):
    0~255         바이트
    256~259       특수 토큰 <pad> <bos> <eos> <sep>
    260~          seed 토큰 (항상 토큰 하나: SQL 키워드, 테이블·컬럼 이름, 범주형 값)
    seed 다음~    BPE 병합 결과

seed 토큰 목록은 단계마다 달라서 파일로 관리한다 (config.SEED_TOKENS_PATH, 사람이 편집).
학습 결과는 특수 토큰 + seed 토큰 + 병합 규칙을 한 파일(config.TOKENIZER_PATH)에 담는다. seed ID 가
목록 순서로 정해지므로, 학습 당시 목록을 결과 파일에 함께 저장해야 나중에 seed 파일을 고쳐도
기존 모델이 그대로 동작한다. 이 파일 하나로 Python 과 C 추론 엔진이 같은 토큰화를 재현한다.

    python -m src.tokenizer.bpe          # seed 파일 + 학습 쌍 → tokenizer.json, round-trip·미등장 이름 토큰 수 확인
    python -m src.tokenizer.bpe --train-files a.json b.json --seeds s.json --out t.json   # 입력·출력 바꾸기
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.config import VOCAB_SIZE, TRAIN_DIR, HOLDOUT_PATH, SEED_TOKENS_PATH, TOKENIZER_PATH

SPECIAL_BASE = 256
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>"]
SEED_BASE = SPECIAL_BASE + len(SPECIAL_TOKENS)
FORMAT_VERSION = 1

Piece = tuple[str, "int | None"]
Merge = tuple[tuple[int, int], int]


def _validate_seeds(seeds: list[str]) -> None:
    assert len(seeds) == len(set(seeds)), "Seed tokens must be unique."
    for s in seeds:
        assert len(s) >= 2, f"Seed token '{s}' must be at least 2 characters long."
        assert s.strip() == s, f"Seed token '{s}' must not have leading or trailing whitespace."


def _build_pattern(seeds: list[str]) -> re.Pattern:
    alts: list[str] = []
    for s in sorted(seeds, key=len, reverse=True):
        alts.append(r"\b" + re.escape(s) + r"\b")      # 단어 경계로 둘러쌓여 있는 시드 토큰 (' . , * 등으로 둘러쌓여 있는 경우)
    alts.append(r"[^\W\d_]+")                          # 단어문자이면서 숫자와 밑줄이 아닌것, 즉 알파벳만
    alts.append(r"[0-9]")                              # 0~9 까지 한글자만
    alts.append(r"\s")                                 # 공백 문자
    alts.append(r".")                                  # 아무 문자 한글자
    return re.compile("|".join(alts), re.DOTALL)       # re.DOTALL: .이 줄바꿈 문자를 포함한 모든 문자와 매치되도록 함


def load_seed_tokens(path: str | Path = SEED_TOKENS_PATH) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["seed_tokens"]


# ---------------------------------------------------------------------------
# 학습 보조 (seed 와 무관한 순수 함수)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 토크나이저
# ---------------------------------------------------------------------------

class Tokenizer:
    """seed 토큰 목록 + 병합 규칙. 병합 규칙이 비어 있으면 seed + 바이트 단위로만 인코딩한다."""

    def __init__(self, seed_tokens: list[str], merges: list[Merge] | None = None):
        _validate_seeds(seed_tokens)
        self.seed_tokens = list(seed_tokens)
        self.seed_to_id = {s: SEED_BASE + i for i, s in enumerate(self.seed_tokens)}
        self.merge_base = SEED_BASE + len(self.seed_tokens)
        self.merges: list[Merge] = list(merges or [])
        self._pattern = _build_pattern(self.seed_tokens)
        self.id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for s, sid in self.seed_to_id.items():
            self.id_to_bytes[sid] = s.encode("utf-8")
        for (a, b), new_id in self.merges:
            self.id_to_bytes[new_id] = self.id_to_bytes[a] + self.id_to_bytes[b]

    @property
    def vocab_size(self) -> int:
        """실제로 쓰는 ID 개수 (= 마지막 병합 ID + 1)."""
        return self.merge_base + len(self.merges)

    # --- 사전 분할 -----------------------------------------------------------

    def pre_tokenize(self, text: str) -> list[Piece]:
        """text → (조각, seed ID 또는 None). seed 토큰은 단어 경계 안에서 통째로 한 조각이 된다."""
        pieces: list[Piece] = []
        pos = 0
        for m in self._pattern.finditer(text):
            assert m.start() == pos, f"Unexpected gap in text at position {pos}."
            s = m.group(0)
            pieces.append((s, self.seed_to_id.get(s)))
            pos = m.end()
        assert pos == len(text), f"Unexpected gap in text at position {pos}."
        return pieces

    # --- 인코딩 / 디코딩 -----------------------------------------------------

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, sid in self.pre_tokenize(text):
            if sid is not None:
                ids.append(sid)
            else:
                syms = list(piece.encode("utf-8"))
                for pair, new_id in self.merges:
                    syms = merge_word(syms, pair, new_id)
                ids.extend(syms)
        return ids

    def decode(self, ids: list[int]) -> str:
        """특수 토큰은 태그 문자열(<bos> 등)로 복원한다. 모르는 ID 는 KeyError, 깨진 UTF-8 은 UnicodeDecodeError."""
        parts: list[str] = []
        buf = bytearray()
        for i in ids:
            if SPECIAL_BASE <= i < SEED_BASE:
                if buf:
                    parts.append(bytes(buf).decode("utf-8"))
                    buf = bytearray()
                parts.append(SPECIAL_TOKENS[i - SPECIAL_BASE])
            else:
                buf.extend(self.id_to_bytes[i])
        if buf:
            parts.append(bytes(buf).decode("utf-8"))
        return "".join(parts)

    # --- 학습 ----------------------------------------------------------------

    @classmethod
    def train(cls, corpus: list[str], seed_tokens: list[str], vocab_size: int = VOCAB_SIZE) -> "Tokenizer":
        """vocab_size 를 채울 때까지 가장 많이 나온 바이트 쌍을 병합한다 (seed 토큰 조각은 병합 대상 아님)."""
        tok = cls(seed_tokens)
        freqs: Counter[str] = Counter()
        for text in corpus:
            for piece, sid in tok.pre_tokenize(text):
                if sid is None:
                    freqs[piece] += 1
        word_syms: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in freqs}
        merges: list[Merge] = []
        for step in range(vocab_size - tok.merge_base):
            pair_counts = get_pair_counts(word_syms, freqs)
            if not pair_counts:
                break
            best_count = max(pair_counts.values())
            best_pair = min(p for p, c in pair_counts.items() if c == best_count)  # 동점이면 항상 같은 쌍이 뽑히도록 사전순 최솟값으로 고정
            new_id = tok.merge_base + step
            for w in word_syms:
                word_syms[w] = merge_word(word_syms[w], best_pair, new_id)
            merges.append((best_pair, new_id))
        return cls(seed_tokens, merges)

    # --- 저장 / 불러오기 -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "special_base": SPECIAL_BASE, "special_tokens": SPECIAL_TOKENS,
            "seed_base": SEED_BASE, "seed_tokens": self.seed_tokens,
            "merge_base": self.merge_base, "merges": [[a, b, new_id] for (a, b), new_id in self.merges],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path, legacy_seed_path: str | Path = SEED_TOKENS_PATH) -> "Tokenizer":
        """tokenizer.json 을 읽는다. 예전 형식(병합 규칙만 있는 bpe_merges.json 리스트)이면 seed 목록은
        legacy_seed_path 에서 읽는다 — 이전 1단계 실행 결과(runs/ 의 bpe_merges.json)를 계속 평가하기 위해서다."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls(load_seed_tokens(legacy_seed_path), [((a, b), new_id) for a, b, new_id in data])
        if data["special_tokens"] != SPECIAL_TOKENS or data["special_base"] != SPECIAL_BASE:
            raise ValueError(f"{path}: 특수 토큰 배치가 코드와 다름 ({data['special_tokens']})")
        tok = cls(data["seed_tokens"], [((a, b), new_id) for a, b, new_id in data["merges"]])
        if tok.merge_base != data["merge_base"]:
            raise ValueError(f"{path}: merge_base 불일치 ({data['merge_base']} != {tok.merge_base})")
        return tok


# ---------------------------------------------------------------------------
# 코퍼스
# ---------------------------------------------------------------------------

def load_corpus(files: list[str | Path] | None = None, train_dir: str | Path = TRAIN_DIR) -> list[str]:
    """BPE 코퍼스 = 학습 쌍의 (질문, SQL) 전부. files 를 주면 그 파일들, 아니면 학습 폴더의 *.json
    (dataset.load_train_pairs 와 같은 규칙. bpe 는 torch 없이 돌아야 해서 따로 둔다)."""
    corpus: list[str] = []
    for pf in [Path(f) for f in files] if files else sorted(Path(train_dir).glob("*.json")):
        with open(pf, encoding="utf-8") as f:
            pairs = json.load(f)
        for row in pairs:
            corpus.append(row["question"])
            corpus.append(row["sql"])
    return corpus


def main() -> int:
    ap = argparse.ArgumentParser(description="BPE 토크나이저 학습")
    ap.add_argument("--train-files", nargs="+", default=None, help=f"코퍼스 학습 쌍 파일 (기본: {TRAIN_DIR}/*.json)")
    ap.add_argument("--seeds", default=str(SEED_TOKENS_PATH), help=f"seed 토큰 파일 (기본 {SEED_TOKENS_PATH})")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, help=f"목표 vocab 크기 (기본 {VOCAB_SIZE})")
    ap.add_argument("--out", default=str(TOKENIZER_PATH), help=f"저장 위치 (기본 {TOKENIZER_PATH})")
    ap.add_argument("--holdout", default=str(HOLDOUT_PATH), help=f"미등장 이름 토큰 수 확인용 (기본 {HOLDOUT_PATH})")
    args = ap.parse_args()

    corpus = load_corpus(args.train_files)
    print(f"코퍼스 크기: {len(corpus)}")

    seeds = load_seed_tokens(args.seeds)
    tok = Tokenizer.train(corpus, seeds, args.vocab_size)
    print(f"seed 토큰 {len(seeds)}개, 학습된 병합 수: {len(tok.merges)} / {args.vocab_size - tok.merge_base}")
    tok.save(args.out)
    print(f"저장: {args.out}")

    bad = 0
    for text in corpus:
        restored = tok.decode(tok.encode(text))
        if restored != text:
            bad += 1
            if bad <= 5:
                print(f"ROUNDTRIP FAIL: {text!r} -> {restored!r}")
    print(f"round-trip 실패: {bad} / {len(corpus)}")

    holdout_path = Path(args.holdout)
    if holdout_path.exists():
        with open(holdout_path, encoding="utf-8") as f:
            holdout = json.load(f)
        names = [r["name"] for r in holdout["customers"]["heldout_rows"]]
        names += [r["item_name"] for r in holdout["items"]["heldout_rows"]]
        counts = [len(tok.encode(n)) for n in names]
        print(f"미등장 이름 {len(names)}개 평균 토큰 수: {sum(counts) / len(counts):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
