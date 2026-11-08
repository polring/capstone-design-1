# 역할: 미리 만든 사전으로 사용자 표현을 바꾸고 이름·숫자·검색어를 보호한다.
# 사용처: run.py의 선택형 입력 전처리에서 BPE 토큰화 전에 질문을 정규화할 때 사용한다.

"""Optional, label-free input normalization using a precomputed WordNet map.

Runtime needs only Python's standard library and the map JSON, not WordNet/NLTK.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WORD = re.compile(r"(?<![\w])[a-z]+(?:'[a-z]+)?(?![\w])", re.I)
QUOTED = re.compile(r'''"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)'(?:[^']|'')*'(?!\w)''')
NUMBER = re.compile(r"(?<!\w)\d+(?:[./:\-]\d+)*(?!\w)")
LIKE_VALUES = [
    re.compile(r'\b(?:start(?:s|ing)?|begin(?:s|ning)?|end(?:s|ing)?)\s+(?:with|in)\s+'
               r'(?:(?:the\s+)?(?:letters?|characters?|text|string|fragment|prefix|suffix)\s+)?(?P<value>[a-z]+)\b', re.I),
    re.compile(r'\bcontain(?:s|ing)?\s+(?:(?:the\s+)?(?:text|string|substring|fragment|letters?|characters?)\s+)?(?P<value>[a-z]+)\b', re.I),
    re.compile(r'\b(?:prefix|suffix)\s+(?:of\s+)?(?P<value>[a-z]+)\b', re.I),
    re.compile(r'\b(?:letters?|characters?|substring)\s+(?P<value>[a-z]+)\b', re.I),
    re.compile(r'(?<!\w)(?P<value>[%*][a-z]+[%*]?|[a-z]+[%*])(?!\w)', re.I),
]
LIKE_VALUES_V3 = [
    re.compile(r'\b(?:start(?:s|ing)?|begin(?:s|ning)?|end(?:s|ing)?|finish(?:es|ing)?)\s+'
               r'(?:(?:with|in)\s+)?(?:(?:the\s+)?(?:letters?|characters?|text|string|fragment)\s+)?(?P<value>[a-z]+)\b', re.I),
    re.compile(r'\b(?P<value>[a-z]+)\s+(?:anywhere|somewhere)\s+(?:in|within)\b', re.I),
    re.compile(r'\b(?P<value>[a-z]+)\s+at\s+(?:the\s+)?(?:start|end|beginning)\b', re.I),
]


class WordNetNormalizer:
    def __init__(self, data: dict):
        if data.get('format_version') != 1:
            raise ValueError('Unsupported WordNet map format')
        self.mapping = data['mapping']
        self.trained_words = frozenset(data['trained_words'])
        if self.trained_words.intersection(self.mapping):
            raise ValueError('A WordNet map must not replace observed training words')
        for source, (target, distance) in self.mapping.items():
            if not WORD.fullmatch(source) or target not in self.trained_words or not WORD.fullmatch(target):
                raise ValueError(f'Invalid WordNet replacement: {source!r} -> {target!r}')
            if not isinstance(distance, int) or distance < 0:
                raise ValueError('Invalid WordNet path distance')
        self.metadata = data.get('metadata', {})
        self.phrase_rules = [(rule['id'], re.compile(rule['pattern'], re.I), rule['replacement'])
                             for rule in data.get('phrase_rules', [])]
        self.path = None
        self._values = {}
        for value in data.get('protected_values', []):
            words = [m.group().lower() for m in WORD.finditer(value)]
            if not words:
                continue
            # Only whitespace-separated words are catalogued; punctuation/numbers
            # remain protected by quotes/numeric spans instead of broad matching.
            if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?(?:\s+[a-z]+(?:'[a-z]+)?)*", value, re.I):
                continue
            node = self._values
            for word in words:
                node = node.setdefault(word, {})
            node[None] = True

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        obj = cls(json.loads(path.read_text(encoding='utf-8')))
        obj.path = str(path)
        return obj

    def protected_spans(self, text: str) -> list[tuple[int, int]]:
        spans = [m.span() for pattern in (QUOTED, NUMBER) for m in pattern.finditer(text)]
        if self.metadata.get('protect_like_fragments', False):
            patterns = LIKE_VALUES + (LIKE_VALUES_V3 if self.metadata.get('like_guard_version', 2) >= 3 else [])
            spans.extend(m.span('value') for pattern in patterns for m in pattern.finditer(text))
        tokens = list(WORD.finditer(text))
        for i, first in enumerate(tokens):
            node = self._values
            end = None
            for j in range(i, len(tokens)):
                token = tokens[j]
                if j > i and not text[tokens[j-1].end():token.start()].isspace():
                    break
                node = node.get(token.group().lower())
                if node is None:
                    break
                if None in node:
                    end = token.end()
            if end is not None:
                spans.append((first.start(), end))
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    def normalize(self, question: str) -> tuple[str, list[dict]]:
        """Return transformed text and original-offset changes; never reads SQL."""
        protected = self.protected_spans(question)
        changes, pieces, cursor, span_index = [], [], 0, 0
        candidates = []
        for match in WORD.finditer(question):
            replacement = self.mapping.get(match.group().lower())
            if replacement is not None:
                target, distance = replacement
                candidates.append((match.start(), match.end(), target, distance, 'wordnet'))
        for rule_id, pattern, replacement in self.phrase_rules:
            for match in pattern.finditer(question):
                target = match.expand(replacement)
                if target.lower() != match.group().lower():
                    candidates.append((match.start(), match.end(), target, None, rule_id))
        # Longest phrase wins at the same position. Each original span is
        # replaced once; generated text never enters a second substitution pass.
        for start, end, target, distance, rule_id in sorted(candidates, key=lambda c:(c[0], -c[1], c[4])):
            if start < cursor:
                continue
            while span_index < len(protected) and protected[span_index][1] <= start:
                span_index += 1
            if span_index < len(protected) and protected[span_index][0] < end:
                continue
            pieces.extend((question[cursor:start], target))
            cursor = end
            changes.append(dict(source=question[start:end], replacement=target, start=start, end=end,
                                distance=distance, similarity=1/(distance+1) if distance is not None else None,
                                rule=rule_id))
        pieces.append(question[cursor:])
        return ''.join(pieces), changes

    def __call__(self, question: str) -> str:
        return self.normalize(question)[0]
