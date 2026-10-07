"""
name_swap.py — 이름 교체 실험용 학습 데이터 생성

학습 쌍 중 WHERE 조건이 이름(customers.name / items.item_name)인 쌍의 일부를, 질문과 SQL 양쪽에서
같은 가짜 이름으로 바꾼다. 모델이 이름 목록을 외우는 대신 질문에서 이름을 옮겨 적도록 만들려는 실험이다
(진행 보고서 2026-10-01: 처음 보는 이름 정답률 0~3%).

    python -m src.data.stage1.name_swap --ratio 0.4      # → data_raw/stage1/name_swap_40/train_pairs.json
    python -m src.train --train-file data_raw/stage1/name_swap_40/train_pairs.json --run-name name_swap_40
    python -m src.train                           # 기본값: epoch 마다 새로 바꾸기 (--name-swap-ratio 0.4)

- 가짜 이름: 실제 이름 목록(db_gen.GIVEN_NAMES / CATEGORY_NOUNS)으로 학습한 글자 단위 Markov chain 으로
  생성해 실제 이름처럼 보이게 한다. 생김새로 진짜/가짜를 구분해 "진짜처럼 생기면 외운 목록에서 고른다"는
  전략을 배우지 못하게 하려는 것이다. DB 의 모든 이름(holdout 포함), 이름 풀, 학습 질문에 나오는 단어,
  실제 영단어(WordNet 표제어와 그 -s/-es/-ed/-ing/-er/-ly 변화형)와 겹치는 것은 버리고, 쌍마다 서로 다른
  이름을 쓴다.
- 영단어 목록: 기본은 저장소의 data/wordnet/english_words.txt (WordNet 3.0 표제어, 한 줄에 한 단어).
  NLTK 의 wordnet.zip 도 표준 라이브러리 zipfile 로 직접 읽을 수 있다. 다른 목록은 --wordlist 로 지정한다.
- 질문 속 이름 위치는 question_gen.literal_in_question 과 같은 규칙(단어 경계, 뒤의 's/s/es 허용)으로 찾고,
  바꾼 뒤 같은 함수로 "새 이름은 있고 옛 이름은 없다"를 확인한다. 확인에 실패한 쌍은 바꾸지 않는다.
- 바꾼 쌍에는 orig_question / orig_sql 을 남긴다. train.py 는 검증셋을 orig_sql 기준으로 떼고 검증에는
  원래 쌍을 쓰므로, 검증셋이 기준 실행과 같아진다.
- 출력은 config.RAW_DIR (커밋 안 함, 학습 폴더 밖) — 학습 데이터나 BPE 코퍼스에 자동으로 섞이지 않는다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from src import config
from src.data.dataset import load_train_pairs
from src.data import db_gen
from src.data.stage1.question_gen import literal_in_question

NAME_SQL = re.compile(r"WHERE (name|item_name) = '([^']+)'")
ENGLISH_SUFFIXES = (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                    ("er", ""), ("er", "e"), ("ly", ""))
NAME_POOLS = {
    "name": list(db_gen.GIVEN_NAMES),
    "item_name": [n for nouns in db_gen.CATEGORY_NOUNS.values() for n in nouns],
}


class MarkovNameGenerator:
    """글자 단위 n-gram Markov chain. 학습 이름과 비슷한 철자 패턴의 새 이름을 만든다."""

    def __init__(self, names: list[str], order: int = 2):
        self.order = order
        self.table: dict[str, Counter] = defaultdict(Counter)
        for n in names:
            padded = "^" * order + n + "$"
            for i in range(len(padded) - order):
                self.table[padded[i:i + order]][padded[i + order]] += 1

    def sample(self, rng: random.Random, min_len: int = 3, max_len: int = 10) -> str | None:
        state, out = "^" * self.order, ""
        while len(out) <= max_len:
            chars, weights = zip(*self.table[state].items())
            c = rng.choices(chars, weights)[0]
            if c == "$":
                return out if len(out) >= min_len else None
            out += c
            state = state[1:] + c
        return None


def load_english_words(path: str | Path) -> set[str]:
    """WordNet zip(index.noun/verb/adj/adv 의 표제어) 또는 한 줄에 한 단어인 텍스트 파일에서 영단어를 읽는다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"영단어 목록 없음: {path} (--wordlist 로 지정)")
    words: set[str] = set()
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if re.search(r"index\.(noun|verb|adj|adv)$", name):
                    for line in z.read(name).decode("latin-1").splitlines():
                        w = line.split(" ", 1)[0]
                        if not line.startswith(" ") and w.isalpha():
                            words.add(w.lower())
    else:
        words = {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}
    return words


def is_english(word: str, english: set[str]) -> bool:
    """표제어이거나, 흔한 변화 어미를 떼면 표제어가 되는 단어 (dogs, parked, running 등)."""
    if word in english:
        return True
    return any(word.endswith(suf) and word[:-len(suf)] + rep in english for suf, rep in ENGLISH_SUFFIXES)


def replace_name(question: str, old: str, new: str) -> str:
    """literal_in_question 과 같은 경계 규칙으로 old 를 찾아 new 로 바꾼다 (뒤의 's/s/es 는 유지)."""
    return re.sub(rf"\b{re.escape(old)}(?=(?:'?s|es)?\b)", new, question)


class NameSwapper:
    """가짜 이름 생성기와 금지 단어 목록을 한 번 만들어 두고, swap() 을 부를 때마다 새로 바꾼 학습 쌍을 만든다.
    고정 파일(build)과 학습 중 epoch 마다 다시 바꾸기(train.py --name-swap-ratio)가 같은 로직을 쓴다."""

    def __init__(self, pairs: list[dict], english: set[str], holdout_path: str | Path = config.HOLDOUT_PATH):
        # 가짜 이름이 피해야 할 단어: DB·holdout 의 모든 이름, 이름 풀, 학습 질문에 나오는 모든 단어
        banned = {w for pool in NAME_POOLS.values() for w in pool}
        holdout = json.load(open(holdout_path, encoding="utf-8"))
        banned |= {r["name"] for r in holdout["customers"]["heldout_rows"]}
        banned |= {r["item_name"] for r in holdout["items"]["heldout_rows"]}
        for p in pairs:
            banned |= set(re.findall(r"[a-z]+", p["question"]))
        self.base_banned = banned
        self.english = english
        self.gens = {col: MarkovNameGenerator(pool) for col, pool in NAME_POOLS.items()}

    def swap(self, pairs: list[dict], ratio: float, rng: random.Random) -> tuple[list[dict], dict]:
        """이름 조건 쌍마다 확률 ratio 로 가짜 이름으로 바꾼다. 한 번 호출 안에서는 쌍마다 서로 다른 이름."""
        banned = set(self.base_banned)
        out, stats = [], Counter()

        def fresh_name(col: str) -> str:
            for _ in range(10_000):
                n = self.gens[col].sample(rng)
                if not n or n in banned:
                    continue
                if is_english(n, self.english):
                    stats["rejected_english"] += 1
                    continue
                banned.add(n)
                return n
            raise RuntimeError(f"{col}: 새 가짜 이름 생성 실패")

        for p in pairs:
            m = NAME_SQL.search(p["sql"])
            if not m:
                out.append(dict(p, swapped=False))
                continue
            stats["name_pairs"] += 1
            if rng.random() >= ratio:
                out.append(dict(p, swapped=False))
                continue
            col, old = m.groups()
            new = fresh_name(col)
            q = replace_name(p["question"], old, new)
            sql = p["sql"].replace(f"= '{old}'", f"= '{new}'")
            if not literal_in_question(q, new, "text") or literal_in_question(q, old, "text"):
                stats["skipped"] += 1
                out.append(dict(p, swapped=False))
                continue
            stats[f"swapped_{col}"] += 1
            out.append(dict(p, question=q, sql=sql, swapped=True,
                            orig_question=p["question"], orig_sql=p["sql"]))
        stats["total_pairs"] = len(pairs)
        return out, dict(stats)


def build(pairs: list[dict], ratio: float, seed: int, english: set[str],
          holdout_path: str | Path = config.HOLDOUT_PATH) -> tuple[list[dict], dict]:
    """고정 파일용: 전체 학습 쌍을 한 번 바꾼다."""
    return NameSwapper(pairs, english, holdout_path).swap(pairs, ratio, random.Random(seed))


def main() -> int:
    ap = argparse.ArgumentParser(description="이름 교체 실험용 학습 데이터 생성")
    ap.add_argument("--ratio", type=float, default=0.4, help="이름 조건 쌍 중 가짜 이름으로 바꿀 비율")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-files", nargs="+", default=None, help=f"바꿀 학습 쌍 파일들 (기본: {config.TRAIN_DIR}/*.json)")
    ap.add_argument("--holdout", default=str(config.HOLDOUT_PATH), help=f"가짜 이름이 피할 미등장 값 (기본 {config.HOLDOUT_PATH})")
    ap.add_argument("--wordlist", default=str(config.WORDLIST_PATH),
                    help="영단어 목록 (WordNet zip 또는 한 줄에 한 단어인 텍스트 파일)")
    ap.add_argument("--out", default=None, help=f"기본: {config.RAW_DIR}/name_swap_<비율%%>/train_pairs.json")
    args = ap.parse_args()

    out_path = Path(args.out or config.RAW_DIR / f"name_swap_{round(args.ratio * 100)}" / "train_pairs.json")
    assert config.TRAIN_DIR.resolve() not in out_path.resolve().parents, \
        f"{config.TRAIN_DIR} 아래에 두면 학습 데이터와 BPE 코퍼스에 자동으로 섞인다"
    pairs, stats = build(load_train_pairs(args.train_files), args.ratio, args.seed, load_english_words(args.wordlist),
                         args.holdout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"{out_path}: {stats}")
    for p in [p for p in pairs if p["swapped"]][:8]:
        print(f"  {p['orig_question']!r}\n→ {p['question']!r} | {p['sql']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
