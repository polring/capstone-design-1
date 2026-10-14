"""
LLM 학습 질문의 유의어 교체 (epoch 마다, train.py 가 이름 교체 전에 적용)

LLM 이 쓴 학습 질문(source 가 "llm:" 으로 시작)의 일상 단어·구절을 질문 조합기 유사어 사전(data/stage2/composer/synonyms.json)의
대체어로 바꾼다("pull up" → "bring up", "i need" → "i'd like"). 조건·정렬 표현과 단서 단어(protected)는 사전에 없으므로 LLM 이 쓴
표현은 그대로 남고 문장 시작·동사만 바뀐다. 대체어에는 eval 전용 표현의 단어가 없다(compose check 가 검사).
바꾼 문장이 값 검사(question_gen.check_question)나 라벨 규칙(clean_ambiguous.drop_reason)에 걸리면 원래 문장을 쓴다
("find which orders" → "get which orders" 는 L2 에 걸린다). 정답 SQL 은 바꾸지 않는다.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.data.stage2 import compose, sql_ast
from src.data.stage2.clean_ambiguous import drop_reason
from src.data.stage2.question_gen import check_question

LLM_SOURCE_PREFIX = "llm:"


def is_llm(pair: dict) -> bool:
    return str(pair.get("source", "")).startswith(LLM_SOURCE_PREFIX)


class SynonymSwapper:
    def __init__(self, synonyms_path: str | Path = compose.SYNONYMS_PATH):
        self.data = json.loads(Path(synonyms_path).read_text(encoding="utf-8"))
        self.prob = self.data.get("prob", 0.5)  # 조합기와 같은 기본 확률
        self._recs: dict[str, dict] = {}

    def _rec(self, pair: dict) -> dict:
        key = pair.get("sql_group", pair["sql"])
        if key not in self._recs:
            self._recs[key] = sql_ast.to_dict(sql_ast.parse(key))
        return self._recs[key]

    def swap(self, pairs: list[dict], prob: float, rng: random.Random) -> tuple[list[dict], dict]:
        """LLM 쌍의 질문에서 사전에 있는 단어·구절마다 확률 prob 로 대체어 하나로 바꾼다. 바뀐 쌍은 synonym_swapped=True."""
        syn = compose.Synonyms(dict(self.data, prob=prob))
        out, stats = [], Counter()
        for p in pairs:
            if not is_llm(p):
                out.append(dict(p, synonym_swapped=False))
                continue
            stats["llm_pairs"] += 1
            q = syn.apply(p["question"], rng)
            if q == p["question"]:
                out.append(dict(p, synonym_swapped=False))
                continue
            rec = self._rec(p)
            if check_question(q, rec) or drop_reason({"question": q, "sql": p["sql"], "sql_group": rec["sql"]}):
                stats["rejected"] += 1
                out.append(dict(p, synonym_swapped=False))
                continue
            stats["swapped"] += 1
            out.append(dict(p, question=q, synonym_swapped=True))
        return out, dict(stats)
