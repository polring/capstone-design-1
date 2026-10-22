# 역할: 토큰 빈도와 분할 길이에 따라 LIKE 검색 조각 후보를 고르게 뽑는다.
# 사용처: 3단계 학습 중 이름 교체기가 1~3글자 LIKE 조각을 증강할 때 사용한다.

"""Token-aware candidates for stage-3 LIKE augmentation (no model training).

For each fragment length, sample equally among available common-token,
rare-token, and multi-token buckets, then uniformly within the chosen bucket.
Counts come only from the supplied training SQL, including repeated examples.
Candidate strings are lowercase ASCII combinations, independent of eval data.
Only the project's 1..3 character fragments are supported; longer ones skip.
"""
from collections import Counter
from itertools import product
import random
import re
import string

from src.tokenizer.bpe import Tokenizer


class LikeFragmentSampler:
    BUCKETS = ("common", "rare", "multi")
    MAX_LENGTH = 3

    def __init__(self, pairs: list[dict], tokenizer: Tokenizer, *,
                 rare_max_count: int = 100, banned_fragments=()):
        if rare_max_count < 0:
            raise ValueError("rare_max_count must be nonnegative")
        self.tokenizer = tokenizer
        self.rare_max_count = rare_max_count
        self.banned = set(banned_fragments)
        self.token_counts = Counter()
        sql_counts = Counter(p["sql"] for p in pairs)
        self.training_pairs = sum(sql_counts.values())
        self.unique_sql = len(sql_counts)
        for sql, weight in sql_counts.items():
            for token, count in Counter(tokenizer.encode(sql)).items():
                self.token_counts[token] += weight * count
        # A one-token string must occur as a complete vocabulary entry. Only
        # these entries need BPE encoding; all other combinations are multi-token.
        # This avoids running the full merge list for all 17,576 trigrams.
        self._single_tokens = {}
        for data in set(tokenizer.id_to_bytes.values()):
            if 1 <= len(data) <= self.MAX_LENGTH and re.fullmatch(rb"[a-z]+", data):
                fragment = data.decode("ascii")
                ids = tokenizer.encode(fragment)
                if len(ids) == 1:
                    self._single_tokens[fragment] = ids[0]
        self._pools = {}

    def pools(self, length: int) -> dict[str, tuple[str, ...]]:
        """Deterministic, cached candidate buckets; no epoch RNG is consumed."""
        if length not in self._pools:
            pools = {bucket: [] for bucket in self.BUCKETS}
            if 1 <= length <= self.MAX_LENGTH:
                for chars in product(string.ascii_lowercase, repeat=length):
                    fragment = "".join(chars)
                    if fragment in self.banned:
                        continue
                    token = self._single_tokens.get(fragment)
                    if token is None:
                        bucket = "multi"
                    else:
                        bucket = "rare" if self.token_counts[token] <= self.rare_max_count else "common"
                    pools[bucket].append(fragment)
            self._pools[length] = {key: tuple(values) for key, values in pools.items()}
        return self._pools[length]

    def sample(self, length: int, question: str, used: set[str],
               rng: random.Random) -> tuple[str, str] | None:
        # The current fragment and unrelated existing words must not be chosen.
        excluded = used | set(re.findall(r"\b[a-z]+\b", question.lower()))
        pools = self.pools(length)
        available = [key for key, values in pools.items() if any(f not in excluded for f in values)]
        if not available:
            return None
        bucket = rng.choice(available)
        candidates = pools[bucket]
        for _ in range(32):
            fragment = rng.choice(candidates)
            if fragment not in excluded:
                return fragment, bucket
        # Preserve uniform choice even if almost every candidate is blocked.
        return rng.choice([f for f in candidates if f not in excluded]), bucket

    def describe(self) -> dict:
        return {
            "strategy": "balanced", "version": 1,
            "frequency_source": "training SQL only, with pair multiplicity",
            "training_pairs": self.training_pairs, "unique_sql": self.unique_sql,
            "rare_max_count": self.rare_max_count,
            "supported_lengths": list(range(1, self.MAX_LENGTH + 1)),
            "candidate_source": "lowercase ASCII combinations; exact heldout literals excluded",
            "bucket_counts": {str(n): {key: len(values) for key, values in self.pools(n).items()}
                              for n in range(1, self.MAX_LENGTH + 1)},
        }
