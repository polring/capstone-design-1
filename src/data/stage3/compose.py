# 역할: 3단계 SQL 구조를 문장 틀과 표현 사전으로 조합해 영어 질문을 만든다.
# 사용처: 3단계 학습 질문과 표현·구조별 평가 질문을 생성하고 검증할 때 사용한다.

"""
3단계 질문 조합기: SQL 레코드(sql_gen 출력) → 영어 질문 한 문장

SQL 은 sql_gen 이 만들고, 질문은 SQL 구조(select, conds, conn, order, limit)를 조각별로 문장으로 바꿔 조합한다.
LLM 이 문장을 쓰지 않으므로 라벨이 만들 때부터 규칙대로 붙는다(3단계 라벨 규칙:
docs/text-to-sql-label-rules.md).

    문장 틀 (data/stage3/composer/frames.json)  : 역할(star / ident / own_id / exists / attr)별 문형
    표현 사전 (data/stage3/composer/lexicon.json): (SQL 조각, 위치) → 같은 뜻의 표현 목록
    유사어 사전 (data/stage3/composer/synonyms.json): 틀 문장의 일상 단어 → 같은 뜻의 대체어 (단서 단어는 바꾸지 않음)
    값                                          : SQL 리터럴을 질문에 그대로 복사 (price 는 20 / 20.00 / $20)

조합 순서: 역할·수(단수/복수) 결정 → 틀 선택 → 틀의 일상 단어를 유사어로 바꾸기 → 조건마다 위치(명사 앞/뒤/명사구 전체)와 표현 선택
→ 정렬·LIMIT 표현(최상급 / with the lowest … / top n … by / 문장 끝 ", lowest stock first, just 10",
  LIMIT 없는 정렬은 문장 끝에 ", sorted by …" 또는 쉼표 없이 " sorted by …")
→ 문법 후처리(a/an, 공백, 소문자).
난수는 rng_for(seed, epoch, sql) 로 만들어 같은 설정이면 같은 문장, epoch 마다 다른 문장이 나온다.
같은 컬럼의 값 나열(IN)은 질문에 아무 순서로나 나열하고, 학습 쌍의 SQL 은 그 순서를 따른다(question_order_sql, 라벨 규칙 L8).
서로 다른 컬럼의 OR 는 조건을 모두 명사 뒤에 두고 "or" 로 잇는다 ("items in kitchen or under $10").
SQL 레코드의 sql(정규형)은 SQL 의 정체(분할·겹침 검사)로만 쓴다.

split: train 조합기는 split=eval 표시가 된 틀·표현을 쓰지 않는다(처음 보는 표현 평가용으로 남긴다).
eval 조합기는 eval 틀·표현이 있으면 그것을 먼저 쓴다.
평가셋마다 쓰는 조합기 (COMPOSER_SPLIT): indist / struct / holdout 은 train 조합기로 질문을 만들어 처음 보는
SQL·구조·값만 재고, 처음 보는 표현은 phrase 평가가 따로 잰다. phrase 는 indist SQL 을 train 조합기로 조합하되
eval 자리 하나(틀 / 조건 키 하나 / 정렬 키 하나)에서만 eval 표현을 쓴다(compose_phrase). 그래서 오답을 그 표현 탓으로
돌릴 수 있다. eval 명사(patrons 등)는 테이블을 정하는 단어라 학습에 없으면 원리상 맞힐 수 없어 phrase 에 쓰지 않는다.

사용법 (저장소 루트에서)
    python -m src.data.stage3.compose sample --n 20               # 학습 SQL 에서 예시 출력
    python -m src.data.stage3.compose sample --n 20 --split eval
    python -m src.data.stage3.compose check                       # 커버리지·값 복사·사전 충돌·eval 누출·결정성
    python -m src.data.stage3.compose dump --n 300 --out data_raw/stage3/compose_pilot.json   # 검수용 표본
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

# 기존 호출부에서 사용하는 공통 조합기 이름도 이 모듈에서 계속 제공한다.
from src.data.stage2.compose import (
    Composer as BaseComposer, ComposeError,
    Unit, is_wildcard, is_value_key,
    multi_exists_ok, role_of, frame_has_tail,
    entry_family, _words, finish, literal_problems, _value_pos, value_keys,
    lexicon_conflicts,
)
from src.config import stage_paths
from src.data import db_gen
from src.data.stage3 import schema, sql_ast
from src.data.stage3.clean_ambiguous import drop_reason
from src.data.stage3.sql_ast import like_mode, Cond, Query, from_dict, to_dict


STYLE_WEIGHTS = {"common": .70, "conversational": .20, "contextual": .10}


def infer_style(text):
    text = text.strip().lower()
    if re.match(r"(?:for (?:an?|my|our) |i(?: am|'m) (?:working|reviewing|preparing|checking)|we (?:are|need)|as part of)", text):
        return "contextual"
    if re.match(r"(?:can|could|would|will) you\b|(?:hey|hi|any chance|i'd like|i would like|let's|let me)\b", text):
        return "conversational"
    return "common"


def choose_styled(entries, rng, style_of, weights=None):
    """Choose a nonempty style bucket first, then uniformly within that style.

    If a role lacks a style, renormalize the configured probabilities over the
    available styles rather than inventing or mixing incompatible templates.
    """
    buckets = {}
    for entry in entries:
        style = style_of(entry)
        if style not in STYLE_WEIGHTS:
            raise ValueError(f"Unknown template style {style!r}")
        buckets.setdefault(style, []).append(entry)
    if not buckets:
        raise ValueError("No template entries")
    styles = list(buckets)
    probabilities = weights or STYLE_WEIGHTS
    style = rng.choices(styles, weights=[probabilities[s] for s in styles], k=1)[0]
    return rng.choice(buckets[style])


PATHS = stage_paths(3)
COMPOSER_DIR = PATHS.data_dir / "composer"
FRAMES_PATH = COMPOSER_DIR / "frames.json"
LEXICON_PATH = COMPOSER_DIR / "lexicon.json"
SYNONYMS_PATH = COMPOSER_DIR / "synonyms.json"
AGGREGATE_TEMPLATES_PATH = COMPOSER_DIR / "advanced.json"
SPLITS = ("train", "eval")
# SQL 파일(분할)마다 질문을 만드는 조합기. 처음 보는 표현은 phrase 평가(PHRASE_SOURCE 의 SQL + eval 조합기)만 잰다
COMPOSER_SPLIT = {"train": "train", "eval_indist": "train", "eval_struct": "train", "eval_holdout": "train"}
PHRASE_SOURCE = "eval_indist"
EVAL_PHRASE_SETS = ("phrase", "advanced_phrase")  # eval 표현을 쓰는 check 작업 이름
# 명사구 자리: def(the …), bare(한정사 없음), indef(a …), any(any …), nodet(the 없는 def, LIMIT 가능: "top 5 most expensive items"),
# conds(명사 없이 조건만: "order ids for customer 1102"), head(명사구 전체가 값: "order 5312 status?")
NP_KINDS = ("def", "bare", "indef", "any", "nodet", "conds", "head")
PRICE_MONEY_RATIO = 0.4  # {v} 자리의 price 를 $ 표기로 쓰는 비율
DROP_QMARK_RATIO = 0.2   # 끝의 ? 를 빼는 비율 (사용자는 물음표를 자주 생략한다)
TAIL_NO_COMMA_RATIO = 0.5  # 문장 끝 정렬 표현을 쉼표 없이 붙이는 비율 ("clothing items sorted by price"). 명사구로 끝나는 틀,
# sorted / ordered / order by 로 시작하는 표현만 (그 밖은 앞의 값과 붙어 읽힌다: "for product 3064 quantity high to low")
NO_COMMA_SORT = re.compile(r"(sorted|ordered|order by) ")
LIMIT_TAIL_RATIO = 0.2  # LIMIT 을 문장 끝 개수 표현으로 쓰는 비율 ("…, lowest stock first, just 10" / "…, just the top one")
# 값 목록이 고정된 범주형 컬럼 (사전 충돌 검사에서 값이 컬럼을 정하는지 볼 때 쓴다)
CAT_VALUES = {("customers", "city"): db_gen.CITIES, ("customers", "membership"): db_gen.MEMBERSHIPS,
              ("items", "category"): db_gen.CATEGORIES, ("orders", "status"): db_gen.STATUSES}


@dataclass(frozen=True)
class Entry:
    sg: str
    pl: str
    split: str
    ref: str  # 추적용 위치 (예: cond:items.price <:post:2)
    style: str = ""

    def text(self, number: str) -> str:
        return self.sg if number == "sg" else self.pl


def _entry(raw, ref: str) -> Entry:
    if isinstance(raw, str):
        return Entry(raw, raw, "train", ref)
    split = raw.get("split", "train")
    if split not in SPLITS:
        raise ValueError(f"{ref}: split 은 {SPLITS} 중 하나 ({split!r})")
    if "text" in raw:
        return Entry(raw["text"], raw["text"], split, ref, raw.get("style", ""))
    return Entry(raw["sg"], raw["pl"], split, ref, raw.get("style", ""))


_WILDCARD = re.compile(r"[*%]\{v\}|\{v\}[*%]")
# LIKE 표현 중 와일드카드 표현을 고를 확률. 표현 수대로(약 33%) 고르면 따옴표 LIKE 문장이 줄어 학습에 없는 LIKE × ORDER BY 조합에서
# LIKE 조건을 빠뜨린다(2단계 보고서 A-9). 10% 면 사람 작성 와일드카드 질문도 맞히고 이 회귀도 없다
WILDCARD_RATIO = 0.1


# 이 단계에서 사용하는 값 표기·목록 연결·난수 생성 보조 함수
def rng_for(seed: int, epoch: int, sql: str) -> random.Random:
    """SQL 마다 독립된 난수 (문자열 seed 는 실행마다 같은 값이다)."""
    return random.Random(f"{seed}:{epoch}:{sql}")


def _num_text(table: str, col: str, val, money: bool, whole_int: bool) -> str:
    """price 정수 값은 whole_int 면 20, 아니면 20.00 으로 쓴다."""
    if (table, col) in schema.FLOAT_COLUMNS:
        f = float(val)
        s = str(int(f)) if f == int(f) and whole_int else f"{f:.2f}"
        return "$" + s if money else s
    return str(int(val))


def _value_text(
    table: str, unit, val, money: bool, whole_int: bool, rng: random.Random,
    tag: str | None = None, short_words: frozenset[str] = frozenset(),
    wildcard: bool = False,
) -> str:
    """money / whole_int: price 표기 ($ 붙임 / 정수 값을 20 으로). 다른 컬럼은 무시.
    tag 가 있으면 값 대신 tag 를 쓴다($ 와 따옴표는 그대로 붙인다).
    short_words: 따옴표 없이 쓰지 않는 짧은 LIKE 조각 (조합기 문장에 나오는 단어: a, in, to …).
    wildcard: 표현이 조각에 와일드카드를 붙인 것({v}*, *{v}*, %{v}). 기호가 경계를 정하므로 따옴표를 붙이지 않는다."""
    if unit.op == "LIKE" and wildcard:
        return tag or val
    if unit.op == "LIKE":
        # 포함 표현의 3글자 이하 조각은 따옴표 없이 쓰면 영어 단어로 읽힌다 ("with ed in their names").
        # 접두·접미는 바로 앞의 starts with / ends in 이 경계를 정해 주므로 따옴표 없이도 쓴다 (사용자는 "starts with jo",
        # "starts with d" 처럼 쓴다). 조합기 문장의 단어와 같은 조각은 따옴표를 붙인다 ("ending in in", a → an 후처리)
        text = tag or val
        quoted = [f'"{text}"', f"'{text}'"]
        must_quote = len(val) <= 3 and (unit.key.endswith("contains") or val in short_words)
        return rng.choice(quoted if must_quote else [text, *quoted])
    if schema.is_text(table, unit.col):
        return tag or str(val)
    if tag:
        return "$" + tag if money and (table, unit.col) in schema.FLOAT_COLUMNS else tag
    return _num_text(table, unit.col, val, money, whole_int)


# IN 값 목록의 연결 방식과 비율은 이 단계에서 관리한다.
LIST_JOIN_WEIGHTS = {"or": 50, "and": 25, "comma": 10, "slash": 5, "space": 10}


def _join_or(parts: list[str], rng: random.Random, numeric: bool = False) -> str:
    styles = [s for s in LIST_JOIN_WEIGHTS if numeric or s != "space"]
    style = rng.choices(styles, weights=[LIST_JOIN_WEIGHTS[s] for s in styles])[0]
    if style in ("or", "and"):
        if len(parts) == 2:
            return f"{parts[0]} {style} {parts[1]}"
        return ", ".join(parts[:-1]) + rng.choice([f" {style} ", f", {style} "]) + parts[-1]
    return {"comma": ", ", "slash": "/", "space": " "}[style].join(parts)


def _fill(
    template: str, table: str, unit, rng: random.Random, tagger=None,
    short_words: frozenset[str] = frozenset(),
) -> str:
    """값 자리: {v} 는 price 일부만 $, {$v} 는 항상 $, {#v} 는 $ 없이 ("under 20 bucks").
    $ 표기와 정수 표기(20 / 20.00)는 조건 하나 안에서 통일한다 ("between 5 and $103.99" 방지).
    tagger(unit, j) 가 문자열을 주면 j 번째 값 자리에 값 대신 그 문자열을 쓴다 (태그 모드)."""
    tag = tagger or (lambda unit, j: None)
    plain_money = rng.random() < PRICE_MONEY_RATIO
    whole_int = rng.random() < 0.5
    # 값 나열(IN)의 따옴표는 값마다 붙인다 (named "amy or jane" 은 이름 하나로 읽힌다 → named "amy" or "jane")
    quote = ""
    if unit.is_list and (qm := re.search(r"([\"'])(\{[$#]?v\})\1", template)):
        quote, template = qm.group(1), template.replace(qm.group(0), qm.group(2))

    def sub(m: re.Match) -> str:
        mark, second = m.group(1), m.group(2) == "2"
        money = mark == "$" or (mark == "" and plain_money)
        if unit.is_list:
            # IN 값은 질문에 아무 순서로나 나열하고, 학습 쌍의 SQL 은 그 순서를 따른다(question_order_sql, L8). 늘 값 순서로만
            # 나열하면 사용자가 다른 순서로 쓴 질문("vest or radio")을 학습에서 보지 못한다. 태그 모드는 태그 번호가 질문
            # 순서라 섞은 결과를 쓰지 않는다 (난수는 똑같이 써서 태그 없는 문장과 틀·표현이 같게 한다)
            vals = list(enumerate(unit.values))
            shuffled = rng.sample(vals, len(vals))
            if tagger is None:
                vals = shuffled
            return _join_or([quote + _value_text(table, unit, v, money, whole_int, rng, tag(unit, j)) + quote
                             for j, v in vals], rng, numeric=not schema.is_text(table, unit.col))
        j = 1 if second else 0
        wildcard = m.string[m.start() - 1:m.start()] in ("*", "%") or m.string[m.end():m.end() + 1] in ("*", "%")
        return _value_text(table, unit, unit.values[j], money, whole_int, rng, tag(unit, j), short_words, wildcard)
    return re.sub(r"\{([$#]?)v(2?)\}", sub, template)


# 이 단계의 문장 틀과 학습 질문에 적용하는 유의어 증강
class Synonyms:
    """보호된 값 자리 밖의 표현만 명사·구동사 규칙에 맞춰 바꾼다."""
    def __init__(self, data: dict, context: str | None = None):
        self.prob = data.get('prob', .5)
        self.words = dict(data['words'])
        if context == 'advanced':
            self.words.update(data.get('advanced_words', {}))
        self.protected = set(data.get('protected', []))
        self.start_only = set(data.get('start_only', []))
        self.verb_only = set(data.get('verb_only', []))
        self.skip_followed_by = data.get('skip_followed_by', {})
        keys = sorted(self.words, key=len, reverse=True)
        self._pattern = re.compile(r"(?<![\w'])(" + '|'.join(re.escape(k) for k in keys) + r")(?![\w'])")

    def apply(self, text: str, rng: random.Random) -> str:
        def sub(match, first):
            key = match.group(1)
            if key in self.start_only and not (first and match.start() == 0):
                return key
            before = re.search(r'\b([a-z]+)\s+$', match.string[:match.start()])
            if key in self.verb_only and before and before[1] in {
                'a', 'an', 'the', 'my', 'your', 'our', 'their', 'his', 'her', 'its', 'this', 'that'
            }:
                return key
            after = re.match(r'\s+([a-z]+)\b', match.string[match.end():])
            if after and after[1] in self.skip_followed_by.get(key, []):
                return key
            return rng.choice(self.words[key]) if rng.random() < self.prob else key
        parts = re.split(r'(\{[^}]*\})', text)
        return ''.join(p if p.startswith('{') else self._pattern.sub(lambda m: sub(m, i == 0), p)
                       for i, p in enumerate(parts))


def is_llm(pair):
    return str(pair.get('source', '')).startswith('llm:')


def is_composer(pair):
    return str(pair.get('source', '')).startswith('composer')


def mask_values(question, query):
    values = set()
    for c in (*query.conds, *getattr(query, 'inner_conds', ())):
        raw = c.val if c.op == 'IN' else (c.val, c.val2) if c.op == 'BETWEEN' else (c.val,)
        values.update(v.strip('%') for v in raw if isinstance(v, str) and v.strip('%'))
    patterns = [rf'(?<!\w){re.escape(v)}(?!\w)' for v in sorted(values, key=lambda v: (-len(v), v))]
    patterns.append(r'\d+(?:\.\d+)?')
    slots = []
    def mask(match):
        slots.append(match.group())
        return '{literal_' + str(len(slots) - 1) + '}'
    return re.sub('|'.join(patterns), mask, question), slots


class SynonymSwapper:

    def __init__(self, synonyms_path=None):
        self.data = json.loads(Path(synonyms_path or SYNONYMS_PATH).read_text(encoding='utf-8'))
        self.prob = self.data.get('prob', 0.5)

    def swap(self, pairs, prob, rng, composer_prob=0.0):
        """LLM 쌍은 prob, 조합기 쌍(source composer*)은 composer_prob 확률로 단어마다 바꾼다.
        조합기 문장은 만들 때 틀에 이미 한 번 적용됐으므로, 이건 epoch 마다 다른 대체어를 보여 주는 용도다."""
        if not (0 <= prob <= 1 and 0 <= composer_prob <= 1):
            raise ValueError('Synonym probability must be within 0..1')
        swappers = {}
        for kind, p in (('llm', prob), ('composer', composer_prob)):
            data = {**self.data, 'prob': p}
            swappers[kind] = (Synonyms(data), Synonyms(data, context='advanced'))
        out, stats = ([], Counter())
        for pair in pairs:
            unchanged = dict(pair, synonym_swapped=False)
            kind = 'llm' if is_llm(pair) else 'composer' if composer_prob and is_composer(pair) else None
            if kind is None:
                out.append(unchanged)
                continue
            prefix = '' if kind == 'llm' else 'composer_'
            stats[f'{kind}_pairs'] += 1
            plain, advanced = swappers[kind]
            try:
                query = sql_ast.parse(pair['sql'])
            except (ValueError, KeyError):
                stats[prefix + 'rejected'] += 1
                out.append(unchanged)
                continue
            text, slots = mask_values(pair['question'], query)
            syn = advanced if hasattr(query, 'aggregate') else plain
            question = syn.apply(text, rng)
            for i, value in enumerate(slots):
                question = question.replace('{literal_' + str(i) + '}', value)
            invalid = drop_reason(dict(pair, question=question))
            if question == pair['question']:
                out.append(unchanged)
            elif invalid:
                stats[prefix + 'rejected'] += 1
                out.append(unchanged)
            else:
                stats[prefix + 'swapped'] += 1
                out.append(dict(pair, question=question, synonym_swapped=True))
        return (out, dict(stats))


def short_words(frames: dict, lexicon: dict, synonyms: dict | None) -> frozenset[str]:
    """조합기 문장에 나올 수 있는 3글자 이하 단어(틀·표현·유사어) + a, an, i. 이 단어와 같은 짧은 LIKE 조각은 따옴표를 붙인다."""
    words = {"a", "an", "i"}

    def walk(x):
        if isinstance(x, str):
            words.update(w for w in _words(re.sub(r"\{[^}]*\}", " ", x)) if len(w) <= 3)
        elif isinstance(x, dict):
            for k, v in x.items():
                if k != "_doc":
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(frames)
    walk(lexicon)
    if synonyms:
        walk(synonyms.get("words", {}))
        walk(synonyms.get("advanced_words", {}))
    return frozenset(words)


def synonym_problems(synonyms: dict, frames: dict, lexicon: dict, advanced_templates: dict | None = None) -> list[str]:
    """유사어 사전 검사: 대체어가 단서 단어(protected)를 새로 넣거나 빼면 SELECT 역할·칼럼 판정이 바뀐다.
    eval 전용 틀·표현에만 나오는 단어를 대체어로 쓰면 phrase 평가에 샌다. 학습 틀에 없는 키는 쓰이지 않는다."""
    syn = Synonyms(synonyms, context="advanced" if advanced_templates else None)
    train_words, eval_words, train_frames = set(), set(), []

    def walk(raw):
        e = _entry(raw, "")
        target = eval_words if e.split == "eval" else train_words
        for t in {e.sg, e.pl}:
            target.update(_words(re.sub(r"\{[^}]*\}", " ", t)))
        return e

    for role, group in frames.items():
        if role.startswith("_"):
            continue
        for rows in (group.values() if role == "attr" else [group]):
            for r in rows:
                if walk(r).split == "train":
                    train_frames.append(r["text"])
    for section in ("nouns", "conds", "orders", "limits"):
        for d in lexicon.get(section, {}).values():
            for rows in (d.values() if isinstance(d, dict) else [d]):
                for r in rows:
                    walk(r)
    if advanced_templates:
        for role in ("aggregate", "group", "group_default", "subquery", "subquery_default", "having"):
            for split in ("train", "eval"):
                for text in advanced_templates.get(role, {}).get(split, []):
                    walk({"text": text, "split": split})
                    if split == "train":
                        train_frames.append(text)
        for groups in advanced_templates.get("aggregate_default", {}).values():
            for split, texts in groups.items():
                for text in texts:
                    walk({"text": text, "split": split})
                    if split == "train":
                        train_frames.append(text)
        for groups in advanced_templates.get("aggregates", {}).values():
            for split, texts in groups.items():
                for text in texts:
                    walk({"text": text, "split": split})
    eval_only = {w for w in eval_words - train_words if len(w) > 1}  # 축약 조각(i'd 의 d)은 보지 않는다
    out = []
    for key, alts in syn.words.items():
        if not any(syn._pattern.search(f) and re.search(rf"(?<![\w']){re.escape(key)}(?![\w'])", f)
                   for f in train_frames):
            out.append(f"유사어 키 {key!r} 가 학습 틀에 없음")
        for alt in alts:
            if (_words(key) ^ _words(alt)) & syn.protected:
                out.append(f"유사어 {key!r} → {alt!r}: 단서 단어 {sorted((_words(key) ^ _words(alt)) & syn.protected)}")
            if _words(alt) & eval_only:
                out.append(f"유사어 {key!r} → {alt!r}: eval 전용 단어 {sorted(_words(alt) & eval_only)}")
    return out


TAG_PREFIX = "val"  # 태그 val1, val2, … (토크나이저 seed 토큰은 단어 경계로 찾으므로 영숫자만 쓴다)
_SENTINEL = re.compile("\u00a7(\\d+)\u00a7")  # compose_tagged 가 값 자리에 잠시 넣는 표시


def render_tagged(rec: dict, names: dict[tuple[int, str], str]) -> str:
    """rec 의 SQL 을 그리되 names 에 있는 (조건 위치, val / val2) 의 리터럴은 태그로 쓴다 (sql_ast.Query.render 와 같은 모양)."""
    t = rec["table"]

    def lit(i: int, c: dict, slot: str) -> str:
        tag = names.get((i, slot))
        if tag is None:
            return schema.fmt_literal(t, c["col"], c[slot])
        if c["op"] == "LIKE":
            return "'" + c["val"].replace(like_mode(c["val"])[1], tag, 1) + "'"
        return f"'{tag}'" if schema.is_text(t, c["col"]) else tag

    def cond(i: int, c: dict) -> str:
        if c["op"] == "IN":
            return f"{c['col']} IN (" + ", ".join(
                names.get((i, f"in{j}")) and (f"'{names[(i, f'in{j}')]}'" if schema.is_text(t, c["col"])
                                               else names[(i, f"in{j}")])
                or schema.fmt_literal(t, c["col"], v) for j, v in enumerate(c["val"])) + ")"
        if c["op"] == "BETWEEN":
            return f"{c['col']} BETWEEN {lit(i, c, 'val')} AND {lit(i, c, 'val2')}"
        return f"{c['col']} {c['op']} {lit(i, c, 'val')}"

    parts = [f"SELECT {rec['select']} FROM {t}"]
    if rec["conds"]:
        parts.append("WHERE " + f" {rec['conn'] or 'AND'} ".join(cond(i, c) for i, c in enumerate(rec["conds"])))
    if rec["order"]:
        # Share the SQL renderer's policy and preserve explicitly stored legacy tie orders.
        order_clause = from_dict(rec).render().split(" ORDER BY ", 1)[1].split(" LIMIT ", 1)[0]
        parts.append("ORDER BY " + order_clause)
    if rec["limit"] is not None:
        parts.append(f"LIMIT {rec['limit']}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 조합기
# ---------------------------------------------------------------------------

class Composer(BaseComposer):
    """공통 조합기에 3단계 날짜·문체·고급 SQL 처리를 추가한다."""

    _entry_factory = staticmethod(_entry)
    _fill = staticmethod(_fill)
    _short_words = staticmethod(short_words)

    def __init__(self, frames: dict, lexicon: dict, split: str = "train", synonyms: dict | None = None,
                 wildcard_ratio: float | None = WILDCARD_RATIO, aggregate_templates: dict | None = None):
        super().__init__(frames, lexicon, split, synonyms, wildcard_ratio)
        self.aggregate_templates = aggregate_templates
        self.advanced_synonyms = Synonyms(synonyms, context="advanced") if synonyms else None
        self._contrast_value_keys = value_keys(lexicon)

    @classmethod
    def load(cls, split: str = "train", frames_path: str | Path = FRAMES_PATH,
             lexicon_path: str | Path = LEXICON_PATH, synonyms_path: str | Path | None = SYNONYMS_PATH,
             templates_path: str | Path = AGGREGATE_TEMPLATES_PATH) -> "Composer":
        with open(frames_path, encoding="utf-8") as f:
            frames = json.load(f)
        with open(lexicon_path, encoding="utf-8") as f:
            lexicon = json.load(f)
        synonyms = None
        if synonyms_path:
            with open(synonyms_path, encoding="utf-8") as f:
                synonyms = json.load(f)
        templates = json.loads(Path(templates_path).read_text(encoding="utf-8"))
        return cls(frames, lexicon, split, synonyms, aggregate_templates=templates)

    def _pick(self, entries: list[Entry], rng: random.Random, trace: list[Entry]) -> Entry:
        usable = self._usable(entries)
        if not usable:
            raise ComposeError("쓸 수 있는 표현 없음")
        if self.wildcard_ratio is not None:
            wild = [e for e in usable if is_wildcard(e)]
            if wild and len(wild) < len(usable):
                usable = wild if rng.random() < self.wildcard_ratio else [e for e in usable if not is_wildcard(e)]
        if all(e.ref.startswith("frame:") and e.split == "train" for e in usable):
            e = choose_styled(usable, rng, lambda e: e.style or infer_style(e.sg))
        else:
            e = rng.choice(usable)
        trace.append(e)
        return e

    def compose(self, rec: dict, rng: random.Random) -> tuple[str, list[Entry]]:
        if rec.get("query_kind"):
            return _compose_aggregate(rec, rng, self, self.split)
        return super().compose(rec, rng)

    def compose_contrast(self, queries, seed: int, group: str, count: int = 5):
        """Compose a SQL contrast group using only existing training expressions.

        Shared seeds align wording where the SQL roles permit it. SQL variants
        remain explicit: this does not mutate arbitrary natural-language text.
        Reject the entire candidate if any member cannot receive count distinct,
        rule-consistent questions, rather than silently dropping a contrast arm.
        """
        if self.split != "train":
            raise ValueError("contrast composition requires the training split")
        result = []
        for query in queries:
            rec = to_dict(query)
            seen = set()
            member = []
            for attempt in range(count * 20):
                question, trace = self.compose(rec, rng_for(seed, attempt, group))
                sql = question_order_sql(rec, question)
                if question in seen or any(entry.split != "train" for entry in trace):
                    continue
                if literal_problems(question, rec, self._contrast_value_keys):
                    continue
                if drop_reason({"question": question, "sql": sql}):
                    continue
                seen.add(question)
                member.append({"question": question, "sql": sql,
                               "sql_group": rec["sql"],
                               "template_refs": [entry.ref for entry in trace]})
                if len(member) == count:
                    break
            if len(member) != count:
                raise ComposeError("could not compose every contrast arm")
            result.append(member)
        return result

    def compose_tagged(self, rec: dict, rng: random.Random) -> tuple[str, str, list[Entry]]:
        """값 복사 실험용: 이름·ID·LIKE 조각·숫자 값을 태그(val1, val2, …)로 바꾼 (질문, SQL, 쓴 틀·표현).
        태그 번호는 질문에 나온 순서다. 범주형 값과 LIMIT n 은 그대로 두고, 질문에 숫자 없이 나온 고정값
        (out of stock → stock = 0)도 SQL 에 그대로 남는다."""
        slots: list[tuple[int, str]] = []  # 센티널 번호 → (rec["conds"] 위치, val / val2)

        def tagger(unit: Unit, j: int) -> str | None:
            if schema.kind(rec["table"], unit.col) == "cat":
                return None
            slots.append((unit.idx[0], f"in{j}") if unit.is_list else (unit.idx[0], "val2" if j else "val"))
            return f"\u00a7{len(slots) - 1}\u00a7"

        self._tagger = tagger
        try:
            q, trace = self.compose(rec, rng)
        finally:
            self._tagger = None
        names: dict[tuple[int, str], str] = {}
        for m in _SENTINEL.finditer(q):
            names[slots[int(m.group(1))]] = f"{TAG_PREFIX}{len(names) + 1}"
        q = _SENTINEL.sub(lambda m: names[slots[int(m.group(1))]], q)
        return q, render_tagged(rec, names), trace


# ---------------------------------------------------------------------------
# 검사
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def question_order_sql(rec: dict, question: str) -> str:
    """라벨 규칙 L8: IN 의 값은 질문에 나온 순서로 쓴다(AND·OR 조건 순서는 정규형 그대로). rec 은 정규형 SQL 레코드.
    값을 질문에서 찾지 못하면 ComposeError."""
    q = from_dict(rec)
    scopes = {"conds": q.conds}
    if hasattr(q, "inner_conds"):
        scopes["inner_conds"] = q.inner_conds
    if not any(c.op == "IN" for conditions in scopes.values() for c in conditions):
        return rec["sql"]
    updated = {}
    for scope, conditions in scopes.items():
        conds = []
        for c in conditions:
            if c.op == "IN":
                pos = [_value_pos(question, v) for v in c.val]
                if None in pos or len(set(pos)) != len(pos):
                    raise ComposeError(f"IN 값 위치를 찾지 못함: {question!r} <- {rec['sql']!r}")
                c = Cond(c.col, "IN", tuple(v for _, v in sorted(zip(pos, c.val), key=lambda x: x[0])))
            conds.append(c)
        updated[scope] = tuple(conds)
    return replace(q, **updated).render()


LABELS = {
    "city": "city", "membership": "membership tier", "category": "category", "status": "order status",
    "customer_id": "customer id", "item_id": "item id", "order_date": "order date",
    "price": "price", "stock": "stock level", "quantity": "ordered quantity",
    "name": "customer names", "item_name": "item names", "order_id": "order ids",
}
COMPARISONS = {"=": "equal to", "!=": "not equal to", "<": "less than", ">": "greater than",
               "<=": "at most", ">=": "at least"}


def pick_template(templates, role, split, rng, trace, synonyms=None):
    texts = templates[role][split]
    metadata = templates.get("template_meta", {}).get(role, {}).get(split, [])
    indices = list(range(len(texts)))
    def style(i):
        return metadata[i]["style"] if len(metadata) == len(texts) else infer_style(texts[i])
    index = choose_styled(indices, rng, style, templates.get("style_sampling")) if split == "train" else rng.choice(indices)
    trace.append(Entry(role, role, split, f"advanced:{role}:{split}:{index}", style(index)))
    text = texts[index]
    return synonyms.apply(text, rng) if split == "train" and synonyms is not None else text


def aggregate_text(expression, templates, rng, split, table=None):
    function, col = expression.split("(")
    col = col[:-1]
    if col == "order_date":
        return "earliest order date" if function == "MIN" else "latest order date"
    choices = list(templates["aggregates"][function][split])
    if function == "COUNT" and table and split == "train":
        choices += [f"number of {table}", f"total {table}"]
    return rng.choice(choices).replace("{column}", LABELS.get(col, col))


def _compose_aggregate(record, rng, composer=None, split="train", templates=None):
    composer = composer or Composer.load(split)
    templates = templates or composer.aggregate_templates
    if templates is None:
        templates = composer.aggregate_templates = json.loads(AGGREGATE_TEMPLATES_PATH.read_text(encoding="utf-8"))
    query = from_dict(record)
    trace = []
    synonyms = composer.advanced_synonyms

    def rows(conditions):
        flat = to_dict(Query(query.table, "*", conditions, "AND" if len(conditions) > 1 else None))
        noun = composer._pick(composer.nouns[query.table], rng, trace)
        return composer.noun_phrase(flat, "def", "pl", noun, rng, trace, [])

    phrase = rows(query.conds)
    if query.subquery:
        col, op, inner = query.subquery
        # Explicit projection prevents SELECT * versus identifier ambiguity.
        projection = f"all details of {phrase}" if query.projection == "*" else f"{LABELS[query.projection]} for {phrase}"
        aggregate = aggregate_text(inner, templates, rng, split, query.table)
        comparison = COMPARISONS[op]
        reference = rows(query.inner_conds)
        role_key = "subquery"
        if not query.inner_conds and templates.get("subquery_default") and rng.random() < .5:
            role_key = "subquery_default"
        text = pick_template(templates, role_key, split, rng, trace, synonyms).format(
            projection=projection, column=LABELS[col], comparison=comparison,
            aggregate=aggregate, reference=reference)
        role = "subquery"
    else:
        aggregate = aggregate_text(query.aggregate, templates, rng, split, query.table)
        role = "group" if query.group_by else "aggregate"
        role_key = role
        subject = templates.get("implicit_group_subjects", {}).get(f"{query.table}|{query.aggregate}")
        if role == "group" and subject and templates.get("group_default") and rng.random() < .3:
            role_key = "group_default"
        # 여러 행의 "how much stock / how many units" 는 합계다 (라벨 규칙 L19). 일반 조회 틀은 한 행일 때만 쓴다
        defaults = templates.get("aggregate_default", {})
        default_key = f"{query.table}|{query.aggregate}"
        unique_keys = {schema.TABLE_COLUMNS[query.table][0], "item_name", "name"}
        single_row = any(c.col in unique_keys and c.op == "=" for c in query.conds)
        if role == "aggregate" and default_key in defaults and not single_row and rng.random() < .3:
            text = pick_template(defaults, default_key, split, rng, trace, synonyms).format(rows=phrase)
        else:
            text = pick_template(templates, role_key, split, rng, trace, synonyms).format(
                aggregate=aggregate, rows=phrase, group=LABELS.get(query.group_by, ""), subject=subject)
        text = text.rstrip(" .?!")
        if query.having:
            op, number = query.having
            text += pick_template(templates, "having", split, rng, trace, synonyms).format(
                aggregate=aggregate, comparison=COMPARISONS[op], value=f"{float(number):.6g}")
        if query.order:
            direction = "highest first" if query.order[1] == "DESC" else "lowest first"
            text += f", sort groups by {aggregate}, {direction}"
        if query.limit:
            text += f", return only the first {query.limit} groups"
    text = " ".join(text.split()).lower().rstrip(" .?!") + "?"
    return text, trace


def check(records: dict[str, list[dict]], frames: dict, lexicon: dict, epochs: int = 3, seed: int = 0,
          synonyms: dict | None = None, templates: dict | None = None) -> dict:
    """records: {이름: SQL 레코드}. 사전이 다루는 테이블만 본다. 실패 목록과 통계를 돌려준다.
    이름별 조합기는 COMPOSER_SPLIT 을 따르고(없으면 train), PHRASE_SOURCE 가 있으면 phrase 평가도 함께 본다."""
    fails: list[str] = [f"사전 충돌 {c}" for c in lexicon_conflicts(lexicon)]
    if synonyms:
        advanced_path = COMPOSER_DIR / "advanced.json"
        advanced_templates = templates or (json.loads(advanced_path.read_text(encoding="utf-8")) if advanced_path.exists() else None)
        fails += synonym_problems(synonyms, frames, lexicon, advanced_templates)
    vkeys = value_keys(lexicon)
    stats: dict = {}
    jobs = [(name, rows, COMPOSER_SPLIT.get(name, "train")) for name, rows in records.items()]
    if PHRASE_SOURCE in records:
        jobs.append(("phrase", records[PHRASE_SOURCE], "phrase"))
    if "advanced_indist" in records:
        jobs.append(("advanced_phrase", records["advanced_indist"], "eval"))
    for name, rows, split in jobs:
        comp = Composer(frames, lexicon, "train" if split == "phrase" else split, synonyms,
                        aggregate_templates=templates)
        rows = [r for r in rows if r["table"] in comp.tables]
        errors: Counter = Counter()
        distinct, used = [], Counter()
        no_slot = rule_mismatch = 0
        for r in rows:
            qs = set()
            for ep in range(epochs):
                try:
                    if split == "phrase":
                        got = comp.compose_phrase(r, seed, ep)
                        if got is None:
                            no_slot += 1
                            break
                        q, _, trace, fam = got
                        again = comp.compose_phrase(r, seed, ep)[0]
                    else:
                        q, trace = comp.compose(r, rng_for(seed, ep, r["sql"]))
                        again = comp.compose(r, rng_for(seed, ep, r["sql"]))[0]
                except ComposeError as e:
                    errors[str(e)] += 1
                    break
                if again != q:
                    fails.append(f"[{name}] 결정성 실패 {r['sql']!r}")
                for p in literal_problems(q, r, vkeys):
                    fails.append(f"[{name}] {p}: {q!r} <- {r['sql']!r}")
                try:
                    if drop_reason({"question": q, "sql": question_order_sql(r, q)}):
                        rule_mismatch += 1
                except ComposeError as e:
                    fails.append(f"[{name}] {e}")
                if split == "train" and (leak := [e.ref for e in trace if e.split == "eval"]):
                    fails.append(f"[{name}] eval 표현 사용 {leak}")
                if split == "phrase" and (bad := [e.ref for e in trace
                                                  if e.split == "eval" and entry_family(e.ref) != fam]):
                    fails.append(f"[{name}] 자리 {fam} 밖의 eval 표현 {bad}")
                used.update(e.ref.rsplit(":", 1)[0] if e.ref.startswith("cond") else e.ref for e in trace)
                qs.add(q)
            distinct.append(len(qs))
        for msg, n in errors.most_common():
            fails.append(f"[{name}] 조합 실패 {n}건: {msg}")
        stats[name] = {"n_sql": len(rows), "n_failed": sum(errors.values()), "n_no_eval_slot": no_slot,
                       "avg_distinct_per_sql": round(sum(distinct) / max(1, len(distinct)), 2),
                       "frames_used": sum(1 for k in used if k.startswith("frame:")),
                       "n_rule_mismatch": rule_mismatch}
        # 조합기 문장은 라벨 규칙대로 만든다. 판정(clean_ambiguous)과 어긋나는 문장이 0.1% 를 넘으면 틀·표현과 규칙이 갈린 것이다.
        # eval 표현 세트는 건수만 본다: 검사기 단서는 학습 문장으로만 정하고, 걸린 문항은 eval_build 가 버린다
        if name not in EVAL_PHRASE_SETS and rule_mismatch > 0.001 * max(1, sum(distinct)):
            fails.append(f"[{name}] 라벨 규칙 판정과 어긋난 문장 {rule_mismatch}개 (clean_ambiguous.drop_reason)")
    return {"fails": fails, "stats": stats}


# ---------------------------------------------------------------------------
# 명령행
# ---------------------------------------------------------------------------


def sql_record_paths(sql_dir: str | Path, *, training: bool = False) -> list[Path]:
    """One SQL file per split; query_kind identifies aggregate records."""
    sql_dir = Path(sql_dir)
    names = ("train.json",) if training else ("train.json", "eval_indist.json", "eval_struct.json", "eval_holdout.json")
    return [sql_dir / name for name in names]


def _load_records(paths) -> dict[str, list[dict]]:
    records = {}
    for value in paths:
        path = Path(value)
        rows = json.loads(path.read_text(encoding="utf-8"))
        groups = {
            path.stem: [r for r in rows if not r.get("query_kind")],
            "advanced_" + path.stem.removeprefix("eval_"): [r for r in rows if r.get("query_kind")],
        }
        for name, selected in groups.items():
            if not selected:
                continue
            if name in records:
                raise ValueError(f"SQL 입력 이름이 겹침: {path}")
            records[name] = selected
    return records


def _sample(rows: list[dict], comp, n: int, seed: int, epoch: int) -> list[dict]:
    """조합기가 지원하는 테이블에서 표본을 골라 SQL별 고정 난수로 질문을 만든다."""
    rows = [r for r in rows if r["table"] in comp.tables]
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    out = []
    for r in picked:
        q, trace = comp.compose(r, rng_for(seed, epoch, r["sql"]))
        out.append({"question": q, "sql": r["sql"], "frame": trace[0].ref})
    return out


def build(records: dict[str, list[dict]], comp: Composer, epochs: int, seed: int, out: Path,
          stage1_pairs: str) -> int:
    """질문 생성 원본: SQL 마다 epoch 0..epochs-1 의 문장 (train 조합기). 고정 학습본과 별도로 저장한다.
    지정한 1단계 쌍은 같은 출력 폴더에 사본으로 둔다. 모델 최대 길이를 넘는 쌍은 train.py 가 뺀다.
    sql 은 IN 값을 질문에 나온 순서로 쓴 표기(L8)이고, sql_group 은 정규형 SQL 이다(train.py 가 검증셋을 이것으로 나눈다)."""
    pairs = []
    for rows in records.values():
        for r in rows:
            for ep in range(epochs):
                q = comp.compose(r, rng_for(seed, ep, r["sql"]))[0]
                pairs.append({"question": q, "sql": question_order_sql(r, q), "sql_group": r["sql"],
                              "source": "composer_advanced" if r.get("query_kind") else "composer"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(f"{len(pairs)}개 -> {out}")
    if stage1_pairs:
        # 1단계 쌍은 1단계 규칙으로 정제됐다. 현재 3단계 규칙과 어긋나는 쌍은 사본에서 뺀다
        with open(stage1_pairs, encoding="utf-8") as f:
            s1 = [dict(p, source="stage1") for p in json.load(f)]
        removed = [dict(p, reason=r) for p in s1 if (r := drop_reason(p))]
        s1 = [p for p in s1 if not drop_reason(p)]
        with open(out.parent / "stage1_pairs.json", "w", encoding="utf-8") as f:
            json.dump(s1, f, ensure_ascii=False, indent=1)
        PATHS.raw_dir.mkdir(parents=True, exist_ok=True)
        with open(PATHS.raw_dir / "stage1_pairs_removed.json", "w", encoding="utf-8") as f:
            json.dump(removed, f, ensure_ascii=False, indent=1)
        print(f"1단계 학습 쌍 {len(s1)}개 -> {out.parent / 'stage1_pairs.json'} "
              f"(3단계 라벨 규칙과 어긋난 {len(removed)}개 제외 -> {PATHS.raw_dir / 'stage1_pairs_removed.json'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="3단계 일반 조회·날짜·집계·서브쿼리 질문 조합기")
    ap.add_argument("--sql-dir", default=str(PATHS.sql_dir), help="3단계 SQL 폴더")
    ap.add_argument("command", choices=["sample", "check", "dump", "build"])
    ap.add_argument("--sql", nargs="+", default=None,
                    help="SQL 레코드 파일 (기본: --sql-dir의 일반 조회·집계 계열 모두)")
    ap.add_argument("--templates", default=str(AGGREGATE_TEMPLATES_PATH), help="집계·그룹·서브쿼리 질문 틀")
    ap.add_argument("--frames", default=str(FRAMES_PATH), help=f"문장 틀 (기본 {FRAMES_PATH})")
    ap.add_argument("--lexicon", default=str(LEXICON_PATH), help=f"표현 사전 (기본 {LEXICON_PATH})")
    ap.add_argument("--synonyms", default=str(SYNONYMS_PATH), help=f"유사어 사전, '' 이면 쓰지 않음 (기본 {SYNONYMS_PATH})")
    ap.add_argument("--split", choices=SPLITS, default="train", help="sample·dump: train 표현만 / eval 표현 우선")
    ap.add_argument("--n", type=int, default=20, help="sample·dump: SQL 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0, help="sample·dump: 같은 SQL 의 다른 문장을 보려면 바꾼다")
    ap.add_argument("--epochs", type=int, default=3, help="check: SQL 마다 조합해 볼 epoch 수 / build: SQL 당 문장 수")
    ap.add_argument("--out", default=None, help="dump: 저장할 JSON 파일 (예: data_raw/stage3/compose_pilot.json) / "
                                                f"build: 생성 원본 파일 (기본 {PATHS.raw_dir}/composer/pairs.json)")
    ap.add_argument("--stage1-pairs", default="",
                    help="build: --out 옆에 stage1_pairs.json 으로 복사할 1단계 학습 쌍 ('' 이면 복사 안 함)")
    ap.add_argument("--wildcard-ratio", type=float, default=WILDCARD_RATIO,
                    help=f"sample·dump·build: LIKE 표현 중 와일드카드 표현을 고를 확률 (기본 {WILDCARD_RATIO})")
    args = ap.parse_args(argv)
    sql_dir = Path(args.sql_dir)
    if args.epochs < 1:
        ap.error("--epochs must be positive")
    templates = json.loads(Path(args.templates).read_text(encoding="utf-8"))

    with open(args.frames, encoding="utf-8") as f:
        frames = json.load(f)
    with open(args.lexicon, encoding="utf-8") as f:
        lexicon = json.load(f)
    synonyms = None
    if args.synonyms:
        with open(args.synonyms, encoding="utf-8") as f:
            synonyms = json.load(f)

    if args.command == "check":
        paths = args.sql or sql_record_paths(sql_dir)
        result = check(_load_records(paths), frames, lexicon, args.epochs, args.seed, synonyms, templates)
        for name, st in result["stats"].items():
            print(f"[{name}] SQL {st['n_sql']}개, 실패 {st['n_failed']}, SQL 당 서로 다른 문장 "
                  f"{st['avg_distinct_per_sql']} / {args.epochs}, 쓴 틀 {st['frames_used']}종, 라벨 규칙 판정과 어긋남 {st['n_rule_mismatch']}")
        fails = result["fails"]
        print("통과" if not fails else f"실패 {len(fails)}건\n  - " + "\n  - ".join(fails[:30]))
        return 1 if fails else 0

    if args.command == "build":
        records = _load_records(args.sql or sql_record_paths(sql_dir, training=True))
        out = Path(args.out or PATHS.raw_dir / "composer" / "pairs.json")
        result = build(records, Composer(frames, lexicon, "train", synonyms, args.wildcard_ratio, templates),
                       args.epochs, args.seed, out, args.stage1_pairs)
        # Generated candidates stay outside the frozen training/validation corpus.
        return result

    comp = Composer(frames, lexicon, args.split, synonyms, args.wildcard_ratio, templates)
    rows = [r for rs in _load_records(args.sql or sql_record_paths(sql_dir, training=True)).values() for r in rs]
    samples = _sample(rows, comp, args.n, args.seed, args.epoch)
    if args.command == "sample":
        for s in samples:
            print(f"{s['question']}\n    {s['sql']}")
        return 0
    if not args.out:
        ap.error("dump 는 --out 이 필요하다")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    print(f"{len(samples)}개 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
