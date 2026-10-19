# 역할: 2단계 질문·SQL의 이름과 LIKE 조각을 마르코프 후보로 교체한다.
# 사용처: 2단계 학습에서 이름·IN 값·LIKE 조각을 새로운 값으로 교체할 때 사용한다.

"""2단계 이름·IN·LIKE 증강과 마르코프 후보 생성."""
import random
import re
from collections import Counter
from dataclasses import replace

from src.data.stage1.name_swap import NameSwapper as Stage1NameSwapper, is_english
from src.data.stage2 import sql_ast
from src.data.stage2.sql_ast import like_mode

NAME_COLS = ('name', 'item_name')


def _name_pat(name: str) -> str:
    """1단계 replace_name 과 같은 경계: 단어 경계, 뒤의 's/s/es 허용."""
    return rf"\b{re.escape(name)}(?=(?:'?s|es)?\b)"


def _frag_pat(frag: str) -> str:
    return rf"\b{re.escape(frag)}\b"


def _quoted_pat(frag: str) -> str:
    """따옴표 안의 조각만 ('in' / "in")."""
    f = re.escape(frag)
    return rf"(?<=')(?:{f})(?=')|(?<=\")(?:{f})(?=\")"


def _count(pat: str, q: str) -> int:
    return len(re.findall(pat, q))


def _cut(name: str, mode: str, k: int, rng: random.Random) -> str | None:
    """가짜 이름에서 패턴 방향대로 길이 k 조각을 자른다. 이름 전체와 같은 조각은 만들지 않는다."""
    if mode == "prefix":
        return name[:k] if len(name) > k else None
    if mode == "suffix":
        return name[-k:] if len(name) > k else None
    starts = range(1, len(name) - k)  # 처음·끝이 아닌 위치
    return name[(i := rng.choice(starts)):i + k] if starts else None


class NameSwapper(Stage1NameSwapper):
    reuse_candidates = False
    sql_ast = sql_ast

    def fresh_name(self, col, old, banned, stats, rng):
        for _ in range(10000):
            value = self.gens[col].sample(rng)
            if not value or value in banned:
                continue
            if is_english(value, self.english):
                stats['rejected_english'] += 1
                continue
            banned.add(value)
            return value
        raise RuntimeError(f'{col}: new name generation failed')

    def swap(self, pairs: list[dict], ratio: float, rng: random.Random) -> tuple[list[dict], dict]:
        """이름 조건이 있는 쌍마다 확률 ratio 로 그 쌍의 모든 이름·LIKE 조각을 가짜 값으로 바꾼다.
        예제 간 후보 재사용은 허용하지만 한 예제의 서로 다른 값은 구분한다."""
        banned = set(self.base_banned)
        out, stats = [], Counter()

        def fresh_name(col, old):
            return self.fresh_name(col, old, banned, stats, rng)

        def fresh_frag(col: str, mode: str, k: int, question: str, used: set[str]) -> str | None:
            sampler = getattr(self, "like_sampler", None)
            if sampler is not None:
                selected = sampler.sample(k, question, used, rng)
                if selected is not None:
                    fragment, bucket = selected
                    fragment_buckets[bucket] += 1
                    return fragment
                return None
            for _ in range(1_000):
                n = self.gens[col].sample(rng, min_len=k + 1 if mode != "contains" else k + 2)
                f = n and _cut(n, mode, k, rng)
                if f and f not in used and not _count(_frag_pat(f), question):
                    return f
            return None

        for p in pairs:
            fragment_buckets = Counter()
            if self.reuse_candidates:
                banned = set(self.base_banned)
            try:
                q = self.sql_ast.parse(p["sql"])
            except ValueError:
                out.append(dict(p, swapped=False))
                continue
            scopes = {"conds": list(q.conds)}
            if hasattr(q, "inner_conds"):
                scopes["inner_conds"] = list(q.inner_conds)
            targets = [(scope, i) for scope, conds in scopes.items() for i, c in enumerate(conds)
                       if c.col in NAME_COLS and c.op in ("=", "IN", "LIKE")]
            if not targets:
                out.append(dict(p, swapped=False))
                continue
            stats["name_pairs"] += 1
            if rng.random() >= ratio:
                out.append(dict(p, swapped=False))
                continue

            question, ok = p["question"], True
            occurrences = Counter()
            for scope, i in targets:
                c = scopes[scope][i]
                for value in (c.val if c.op == "IN" else [c.val]):
                    occurrences[(c.col, c.op == "LIKE", value)] += 1
            replacements, edits, checks = {}, [], []
            for scope, i in targets:
                c = scopes[scope][i]
                if c.op in ("=", "IN"):
                    olds = list(c.val) if c.op == "IN" else [c.val]
                    news = []
                    for old in olds:
                        key = (c.col, False, old)
                        if key not in replacements:
                            matches = list(re.finditer(_name_pat(old), p["question"]))
                            if not 1 <= len(matches) <= occurrences[key]:
                                ok = False
                                break
                            new = fresh_name(c.col, old)
                            if _count(_name_pat(new), p["question"]):
                                ok = False
                                break
                            replacements[key] = new
                            edits.extend((m.start(), m.end(), new) for m in matches)
                            checks.append((_name_pat(old), _name_pat(new), len(matches)))
                        news.append(replacements[key])
                    if not ok:
                        break
                    new_val = tuple(news) if c.op == "IN" else news[0]
                else:
                    mode, old = like_mode(c.val)
                    key = (c.col, True, c.val)
                    if key not in replacements:
                        new = fresh_frag(c.col, mode, len(old), p["question"], set(replacements.values()))
                        if new is None:
                            ok = False
                            break
                        # 따옴표 밖에 있는 같은 일반 단어는 바꾸지 않는다.
                        pat = _frag_pat if _count(_frag_pat(old), p["question"]) <= occurrences[key] else _quoted_pat
                        matches = list(re.finditer(pat(old), p["question"]))
                        if not 1 <= len(matches) <= occurrences[key]:
                            ok = False
                            break
                        replacements[key] = new
                        edits.extend((m.start(), m.end(), new) for m in matches)
                        checks.append((pat(old), pat(new), len(matches)))
                    new_val = c.val.replace(old, replacements[key], 1)
                if not ok:
                    break
                scopes[scope][i] = self.sql_ast.Cond(c.col, c.op, new_val, c.val2)
            if ok:
                # 원문 위치를 기준으로 치환해 다른 교체 결과를 다시 바꾸지 않는다.
                edits.sort()
                ok = all(a[1] <= b[0] for a, b in zip(edits, edits[1:]))
                if ok:
                    for start, end, value in reversed(edits):
                        question = question[:start] + value + question[end:]
                    ok = all(_count(pn, question) == count and not _count(po, question)
                             for po, pn, count in checks)
            if not ok:
                stats["skipped"] += 1
                out.append(dict(p, swapped=False))
                continue
            sql = replace(q, **{scope: tuple(conds) for scope, conds in scopes.items()}).render()
            for bucket, count in fragment_buckets.items():
                stats[f"like_bucket_{bucket}"] += count
            for scope, i in targets:
                before = getattr(q, scope)[i]
                stats[f"swapped_{before.col}_{ {'=': 'eq', 'IN': 'in', 'LIKE': 'like'}[before.op]}"] += 1
                if scope == "inner_conds":
                    stats["swapped_inner_conditions"] += 1
            stats["swapped_pairs"] += 1
            out.append(dict(p, question=question, sql=sql, swapped=True,
                            orig_question=p["question"], orig_sql=p["sql"]))
        stats["total_pairs"] = len(pairs)
        return out, dict(stats)
