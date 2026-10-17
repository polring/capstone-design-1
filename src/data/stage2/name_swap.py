"""
name_swap.py — 2단계 이름 교체 (학습 중 epoch 마다, train.py 가 STAGE >= 2 일 때 쓴다)

1단계 이름 교체(src/data/stage1/name_swap.py)는 SQL 의 첫 `WHERE name = '…'` 하나만 바꾼다. 2단계 SQL 에는 한 조건에 나열한
이름 여러 개(IN)와 LIKE 조각이 있어서, 1단계 방식이면 나머지 이름과 LIKE 조각은 학습 내내 DB 의 실제 값이라 모델이 외운다
(2단계 보고서 "첫 2단계 학습": holdout 오답 대부분이 이름 나열의 두 번째 이후 이름). 여기서는 SQL 을 sql_ast 로 파싱해
이름 컬럼(name, item_name)의 조건을 전부 바꾼다.

- `=` / IN 이름: 값마다 서로 다른 가짜 이름(1단계와 같은 Markov 생성기·금지 단어). 값은 SQL 의 같은 자리에서 바꾸고 SQL 을 다시
  정렬하지 않는다. 2단계 학습 SQL 의 IN 값은 질문에 나온 순서(L8)라 자리를 그대로 두면 질문 순서와 SQL 순서가 계속 맞는다.
  (정렬하면 모델이 처음 보는 이름을 알파벳순으로 정렬해야 하는데, 15M 모델은 이를 배우지 못하고 이름 복사까지 망가졌다.)
- LIKE 조각: 길이가 같은 가짜 조각. 가짜 이름에서 패턴 방향대로 자른다(접두 / 접미 / 처음·끝이 아닌 가운데, sql_gen 과 같은
  규칙). 짧은 조각은 영단어와 겹치기 쉬워 영단어 검사는 하지 않고, 원래 질문에 단어로 나오지 않는 조각만 쓴다.
- 질문 속 위치: 이름은 1단계 replace_name 과 같은 경계(뒤의 's/s/es 허용), 조각은 단어 경계만. 바꿀 값마다 질문에 정확히
  한 번 나와야 한다. "in", "the" 처럼 다른 단어와 겹치는 조각은 따옴표 안의 것이 하나면 그것만 바꾼다. 바꾼 뒤 "새 값은 한 번,
  옛 값은 없음"을 확인하고, 한 값이라도 실패하면 그 쌍은 바꾸지 않는다.
- 1단계 쌍(`WHERE name = '…'` 하나)도 같은 경로로 바뀐다. 반환 형식(swapped, orig_question, orig_sql, 통계)은 1단계와 같다.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from src.data.stage1.name_swap import NameSwapper as Stage1NameSwapper
from src.data.stage1.name_swap import is_english
from src.data.stage2 import sql_ast
from src.data.stage2.compose import like_mode

NAME_COLS = ("name", "item_name")


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
    """금지 단어·가짜 이름 생성기는 1단계와 같고, swap() 이 SQL 안의 모든 이름 조건을 바꾼다."""

    def swap(self, pairs: list[dict], ratio: float, rng: random.Random) -> tuple[list[dict], dict]:
        """이름 조건이 있는 쌍마다 확률 ratio 로 그 쌍의 모든 이름·LIKE 조각을 가짜 값으로 바꾼다.
        한 번 호출 안에서는 가짜 이름이 쌍마다 서로 다르다 (조각은 짧아서 겹칠 수 있다)."""
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

        def fresh_frag(col: str, mode: str, k: int, question: str, used: set[str]) -> str | None:
            for _ in range(1_000):
                n = self.gens[col].sample(rng, min_len=k + 1 if mode != "contains" else k + 2)
                f = n and _cut(n, mode, k, rng)
                if f and f not in used and not _count(_frag_pat(f), question):
                    return f
            return None

        for p in pairs:
            try:
                q = sql_ast.parse(p["sql"])
            except ValueError:
                out.append(dict(p, swapped=False))
                continue
            targets = [i for i, c in enumerate(q.conds) if c.col in NAME_COLS and c.op in ("=", "IN", "LIKE")]
            if not targets:
                out.append(dict(p, swapped=False))
                continue
            stats["name_pairs"] += 1
            if rng.random() >= ratio:
                out.append(dict(p, swapped=False))
                continue

            question, conds, ok = p["question"], list(q.conds), True
            checks = []  # (옛 값 패턴, 새 값, 새 값 패턴): 바꾼 뒤 새 값은 한 번, 옛 값은 0번
            for i in targets:
                c = conds[i]
                if c.op in ("=", "IN"):
                    olds = list(c.val) if c.op == "IN" else [c.val]
                    news = [fresh_name(c.col) for _ in olds]
                    for old, new in zip(olds, news):
                        if _count(_name_pat(old), question) != 1:
                            ok = False
                            break
                        question = re.sub(_name_pat(old), new, question)
                        checks.append((_name_pat(old), new, _name_pat(new)))
                    new_val = tuple(news) if c.op == "IN" else news[0]
                else:
                    mode, old = like_mode(c.val)
                    new = fresh_frag(c.col, mode, len(old), question, {n for _, n, _ in checks})
                    if new is None:
                        ok = False
                        break
                    new_val = c.val.replace(old, new, 1)
                    # "in", "the" 같은 조각은 질문의 다른 단어와 겹친다. 따옴표 안의 것 하나만 바꾼다
                    pat = _frag_pat if _count(_frag_pat(old), question) == 1 else _quoted_pat
                    if _count(pat(old), question) != 1:
                        ok = False
                        break
                    question = re.sub(pat(old), new, question)
                    checks.append((pat(old), new, pat(new)))
                if not ok:
                    break
                conds[i] = sql_ast.Cond(c.col, c.op, new_val, c.val2)
            if ok:
                ok = all(_count(pn, question) == 1 and not _count(po, question) for po, _, pn in checks)
            if not ok:
                stats["skipped"] += 1
                out.append(dict(p, swapped=False))
                continue
            sql = sql_ast.Query(q.table, q.select, tuple(conds), q.conn, q.order, q.limit).render()
            for i in targets:
                stats[f"swapped_{q.conds[i].col}_{ {'=': 'eq', 'IN': 'in', 'LIKE': 'like'}[q.conds[i].op]}"] += 1
            stats["swapped_pairs"] += 1
            out.append(dict(p, question=question, sql=sql, swapped=True,
                            orig_question=p["question"], orig_sql=p["sql"]))
        stats["total_pairs"] = len(pairs)
        return out, dict(stats)
