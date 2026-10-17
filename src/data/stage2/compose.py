"""
2단계 질문 조합기: SQL 레코드(sql_gen 출력) → 영어 질문 한 문장

SQL 은 sql_gen 이 만들고, 질문은 SQL 구조(select, conds, conn, order, limit)를 조각별로 문장으로 바꿔 조합한다.
LLM 이 문장을 쓰지 않으므로 라벨이 만들 때부터 규칙대로 붙는다(라벨 규칙 L1~L17:
docs/text-to-sql-label-rules.md).

    문장 틀 (data/stage2/composer/frames.json)  : 역할(star / ident / own_id / exists / attr)별 문형
    표현 사전 (data/stage2/composer/lexicon.json): (SQL 조각, 위치) → 같은 뜻의 표현 목록
    유사어 사전 (data/stage2/composer/synonyms.json): 틀 문장의 일상 단어 → 같은 뜻의 대체어 (단서 단어는 바꾸지 않음)
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
    python -m src.data.stage2.compose sample --n 20               # 학습 SQL 에서 예시 출력
    python -m src.data.stage2.compose sample --n 20 --split eval
    python -m src.data.stage2.compose check                       # 커버리지·값 복사·사전 충돌·eval 누출·결정성
    python -m src.data.stage2.compose dump --n 300 --out data_raw/stage2/compose_pilot.json   # 검수용 표본
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.config import stage_paths
from src.data import db_gen
from src.data.stage2 import schema
from src.data.stage2.clean_ambiguous import drop_reason
from src.data.stage2.sql_ast import Cond, Query, from_dict

PATHS = stage_paths(2)
COMPOSER_DIR = PATHS.data_dir / "composer"
FRAMES_PATH = COMPOSER_DIR / "frames.json"
LEXICON_PATH = COMPOSER_DIR / "lexicon.json"
SYNONYMS_PATH = COMPOSER_DIR / "synonyms.json"
SPLITS = ("train", "eval")
# SQL 파일(분할)마다 질문을 만드는 조합기. 처음 보는 표현은 phrase 평가(PHRASE_SOURCE 의 SQL + eval 조합기)만 잰다
COMPOSER_SPLIT = {"train": "train", "eval_indist": "train", "eval_struct": "train", "eval_holdout": "train"}
PHRASE_SOURCE = "eval_indist"
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


class ComposeError(ValueError):
    """사전·틀에 없는 조합이라 문장을 만들 수 없음."""


@dataclass(frozen=True)
class Entry:
    sg: str
    pl: str
    split: str
    ref: str  # 추적용 위치 (예: cond:items.price <:post:2)

    def text(self, number: str) -> str:
        return self.sg if number == "sg" else self.pl


def _entry(raw, ref: str) -> Entry:
    if isinstance(raw, str):
        return Entry(raw, raw, "train", ref)
    split = raw.get("split", "train")
    if split not in SPLITS:
        raise ValueError(f"{ref}: split 은 {SPLITS} 중 하나 ({split!r})")
    if "text" in raw:
        return Entry(raw["text"], raw["text"], split, ref)
    return Entry(raw["sg"], raw["pl"], split, ref)


_WILDCARD = re.compile(r"[*%]\{v\}|\{v\}[*%]")
# LIKE 표현 중 와일드카드 표현을 고를 확률. 표현 수대로(약 33%) 고르면 따옴표 LIKE 문장이 줄어 학습에 없는 LIKE × ORDER BY 조합에서
# LIKE 조건을 빠뜨린다(2단계 보고서 A-9). 10% 면 사람 작성 와일드카드 질문도 맞히고 이 회귀도 없다
WILDCARD_RATIO = 0.1


def is_wildcard(e: Entry) -> bool:
    """LIKE 조각에 와일드카드를 붙인 표현({v}*, *{v}*, %{v} …)."""
    return bool(_WILDCARD.search(e.sg))


def rng_for(seed: int, epoch: int, sql: str) -> random.Random:
    """SQL 마다 독립된 난수 (문자열 seed 는 실행마다 같은 값이다)."""
    return random.Random(f"{seed}:{epoch}:{sql}")


# ---------------------------------------------------------------------------
# SQL 레코드 해석
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """질문에서 한 덩어리로 표현하는 조건 = rec["conds"] 의 조건 하나. IN 은 값이 여러 개인 Unit 이다(is_list)."""
    key: str             # 사전 키 (예: items.price <, items.item_name LIKE prefix, IN 은 = 키: items.category =)
    col: str
    op: str              # IN 도 "=" (사전 키와 같게). 값 나열인지는 is_list
    values: tuple        # = : (값,), IN: 값들, BETWEEN: (하한, 상한), LIKE: (조각,)
    is_list: bool = False
    idx: tuple = ()      # 이 Unit 이 나온 rec["conds"] 위치


def like_mode(pattern: str) -> tuple[str, str]:
    frag = pattern.strip("%")
    if pattern.startswith("%") and pattern.endswith("%"):
        return "contains", frag
    return ("suffix", frag) if pattern.startswith("%") else ("prefix", frag)


def units_of(rec: dict) -> list[Unit]:
    t, conds = rec["table"], rec["conds"]
    out = []
    for i, c in enumerate(conds):
        if c["op"] == "IN":
            out.append(Unit(f"{t}.{c['col']} =", c["col"], "=", tuple(c["val"]), is_list=True, idx=(i,)))
        elif c["op"] == "LIKE":
            mode, frag = like_mode(c["val"])
            out.append(Unit(f"{t}.{c['col']} LIKE {mode}", c["col"], "LIKE", (frag,), idx=(i,)))
        elif c["op"] == "BETWEEN":
            out.append(Unit(f"{t}.{c['col']} BETWEEN", c["col"], "BETWEEN", (c["val"], c["val2"]), idx=(i,)))
        else:
            out.append(Unit(f"{t}.{c['col']} {c['op']}", c["col"], c["op"], (c["val"],), idx=(i,)))
    return out


def value_key(unit: Unit) -> str:
    """값 전용 사전 키 (라벨 규칙 L16 고정값 표현). 예: items.stock = 0"""
    return f"{unit.key} {unit.values[0]}"


def is_value_key(key: str) -> bool:
    """'테이블.컬럼 연산자 값' (items.stock = 0, orders.status = pending). LIKE 키(… LIKE prefix)는 아니다."""
    parts = key.split(" ", 2)
    return len(parts) == 3 and parts[1] in schema.ALL_OPS and parts[1] != "LIKE"


def multi_exists_ok(rec: dict) -> bool:
    """조건 여러 개(또는 = 가 아닌 조건 하나)인 존재 확인 질문을 붙일 수 있는 SQL 인가 (라벨 규칙 L14).
    SELECT 가 식별 컬럼이고 정렬·LIMIT 이 없어야 한다. = 조건 하나는 L1(조건 컬럼 SELECT) 몫이라 뺀다."""
    conds = rec["conds"]
    single_eq = rec["conn"] is None and len(conds) == 1 and conds[0]["op"] == "="
    return (rec["select"] == schema.IDENT[rec["table"]] and rec["order"] is None and rec["limit"] is None
            and bool(conds) and not single_eq)


def role_of(rec: dict) -> str:
    """SELECT 역할: exists(L1 존재 확인) / star / ident / own_id / attr."""
    t, sel = rec["table"], rec["select"]
    if sel in {c["col"] for c in rec["conds"] if c["op"] == "="}:
        return "exists"
    if sel == "*":
        return "star"
    if sel == schema.IDENT[t]:
        return "ident"
    if sel == schema.OWN_ID.get(t):
        return "own_id"
    return "attr"


def frame_has_tail(text: str) -> bool:
    """명사구 자리 뒤에 단어가 남는 틀인가 ("what category is {np_def} in?" → True)."""
    return re.search(r"[a-z]", text.rsplit("}", 1)[-1]) is not None


def number_of(rec: dict) -> str:
    """질문 명사의 수. 결과가 최대 한 행으로 정해지면(LIMIT 1, 유일 키 = 조건 하나) 단수."""
    if rec["limit"] == 1:
        return "sg"
    conds = rec["conds"]
    if (rec["conn"] is None and len(conds) == 1 and conds[0]["op"] == "="
            and (rec["table"], conds[0]["col"]) in schema.UNIQUE_KEYS):
        return "sg"
    return "pl"


# ---------------------------------------------------------------------------
# 값 표기
# ---------------------------------------------------------------------------

def _num_text(table: str, col: str, val, money: bool, whole_int: bool) -> str:
    """price 정수 값은 whole_int 면 20, 아니면 20.00 으로 쓴다."""
    if (table, col) in schema.FLOAT_COLUMNS:
        f = float(val)
        s = str(int(f)) if f == int(f) and whole_int else f"{f:.2f}"
        return "$" + s if money else s
    return str(int(val))


def _value_text(table: str, unit: Unit, val, money: bool, whole_int: bool, rng: random.Random,
                tag: str | None = None, short_words: frozenset[str] = frozenset(), wildcard: bool = False) -> str:
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


# 값 나열(IN)의 연결: or 만 쓰면 "customers 1224 and 1268", "orders 1620 1988" 을 학습에서 보지 못한다(L9: and 로 나열해도 IN).
# 공백 연결은 숫자 값에만 쓴다("new york" 처럼 값 안에 공백이 있을 수 있다)
LIST_JOIN_WEIGHTS = {"or": 50, "and": 25, "comma": 10, "slash": 5, "space": 10}


def _join_or(parts: list[str], rng: random.Random, numeric: bool = False) -> str:
    styles = [s for s in LIST_JOIN_WEIGHTS if numeric or s != "space"]
    style = rng.choices(styles, weights=[LIST_JOIN_WEIGHTS[s] for s in styles])[0]
    if style in ("or", "and"):
        if len(parts) == 2:
            return f"{parts[0]} {style} {parts[1]}"
        return ", ".join(parts[:-1]) + rng.choice([f" {style} ", f", {style} "]) + parts[-1]
    return {"comma": ", ", "slash": "/", "space": " "}[style].join(parts)


def _fill(template: str, table: str, unit: Unit, rng: random.Random, tagger=None,
          short_words: frozenset[str] = frozenset()) -> str:
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


def entry_family(ref: str) -> str:
    """틀·표현 추적 위치 → phrase 평가의 자리 이름 (frame / noun / cond:<키> / order:<키>)."""
    kind, rest = ref.split(":", 1)
    if kind in ("frame", "noun"):
        return kind
    return f"{kind}:{rest.split(':', 1)[0]}"


class Synonyms:
    """틀 문장의 일상 단어·구절 → 같은 뜻의 대체어. {…} 자리 밖에서만, 긴 구절부터 단어 경계로 찾아
    prob 확률로 대체어 하나로 바꾼다. start_only 키는 문장 맨 앞에서만 바꾼다
    ("tell me …" → "let me know …" 는 되지만 "could you tell me" → "could you let me know" 는 문장이 깨질 수 있다)."""

    def __init__(self, data: dict):
        self.prob = data.get("prob", 0.5)
        self.words: dict[str, list[str]] = data["words"]
        self.protected = set(data.get("protected", []))
        self.start_only = set(data.get("start_only", []))
        keys = sorted(self.words, key=len, reverse=True)
        self._pattern = re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w'])")

    def apply(self, text: str, rng: random.Random) -> str:
        def sub(m: re.Match, first: bool) -> str:
            key = m.group(1)
            if key in self.start_only and not (first and m.start() == 0):
                return key
            return rng.choice(self.words[key]) if rng.random() < self.prob else key
        parts = re.split(r"(\{[^}]*\})", text)
        return "".join(p if p.startswith("{") else self._pattern.sub(lambda m: sub(m, i == 0), p)
                       for i, p in enumerate(parts))


def _words(text: str) -> set[str]:
    """단어 집합. 축약은 나눈다 (what's / whats → what, s)."""
    return set(re.findall(r"[a-z]+", re.sub(r"\bwhats\b", "what s", text.lower())))


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
    return frozenset(words)


def synonym_problems(synonyms: dict, frames: dict, lexicon: dict) -> list[str]:
    """유사어 사전 검사: 대체어가 단서 단어(protected)를 새로 넣거나 빼면 SELECT 역할·칼럼 판정이 바뀐다.
    eval 전용 틀·표현에만 나오는 단어를 대체어로 쓰면 phrase 평가에 샌다. 학습 틀에 없는 키는 쓰이지 않는다."""
    syn = Synonyms(synonyms)
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
        parts.append(f"ORDER BY {rec['order']['col']} {rec['order']['dir']}")
    if rec["limit"] is not None:
        parts.append(f"LIMIT {rec['limit']}")
    return " ".join(parts)


def finish(text: str) -> str:
    """공백·구두점 정리, a/an, 값 1 뒤 단위 단수(1 units → 1 unit), 소문자."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?,])", r"\1", text)
    text = re.sub(r"\ba (?=[aeiou])", "an ", text)
    text = re.sub(r"(?<![\d.])\b1 (unit|piece|dollar|buck)s\b", r"1 \1", text)
    return text.lower()


# ---------------------------------------------------------------------------
# 조합기
# ---------------------------------------------------------------------------

class Composer:
    def __init__(self, frames: dict, lexicon: dict, split: str = "train", synonyms: dict | None = None,
                 wildcard_ratio: float | None = WILDCARD_RATIO):
        """wildcard_ratio: LIKE 표현 중 와일드카드 표현을 고를 확률. None 이면 다른 표현과 똑같이 고른다(표현 수 비율)."""
        if split not in SPLITS:
            raise ValueError(f"split 은 {SPLITS} 중 하나 ({split!r})")
        self.split = split
        self.wildcard_ratio = wildcard_ratio
        self.synonyms = Synonyms(synonyms) if synonyms else None
        self._tagger = None       # compose_tagged 동안만 설정
        self._eval_family = None  # compose_phrase 동안만 설정: 이 자리에서만 eval 표현을 쓴다
        # 틀마다 (항목, 수, 테이블). tables 가 없으면 모든 테이블에 쓴다 ("do we carry" 는 items 만)
        self.frames: dict[str, list[tuple[Entry, str | None, tuple[str, ...] | None]]] = {}
        for role, group in frames.items():
            if role.startswith("_"):
                continue
            items = group.items() if role == "attr" else [(None, group)]
            for sub, rows in items:
                name = f"attr.{sub}" if sub else role
                self.frames[name] = [(_entry(r, f"frame:{name}:{i}"), r.get("number"),
                                      tuple(r["tables"]) if "tables" in r else None) for i, r in enumerate(rows)]
        self.nouns = {t: [_entry(r, f"noun:{t}:{i}") for i, r in enumerate(rows)]
                      for t, rows in lexicon["nouns"].items()}
        self.conds = {k: {pos: [_entry(r, f"cond:{k}:{pos}:{i}") for i, r in enumerate(rows)]
                          for pos, rows in d.items()} for k, d in lexicon["conds"].items()}
        self.orders = {k: {pos: [_entry(r, f"order:{k}:{pos}:{i}") for i, r in enumerate(rows)]
                           for pos, rows in d.items()} for k, d in lexicon["orders"].items()}
        # 정렬 표현 뒤에 붙는 개수 ("lowest stock first, just 10"), {n} 자리. 정렬 컬럼과 무관하다
        self.limit_tails = [_entry(r, f"limit:tail:{i}")
                            for i, r in enumerate(lexicon.get("limits", {}).get("tail", []))]
        self.limit_tails1 = [_entry(r, f"limit:tail1:{i}")
                             for i, r in enumerate(lexicon.get("limits", {}).get("tail1", []))]
        self.short_words = short_words(frames, lexicon, synonyms)

    @classmethod
    def load(cls, split: str = "train", frames_path: str | Path = FRAMES_PATH,
             lexicon_path: str | Path = LEXICON_PATH, synonyms_path: str | Path | None = SYNONYMS_PATH) -> "Composer":
        with open(frames_path, encoding="utf-8") as f:
            frames = json.load(f)
        with open(lexicon_path, encoding="utf-8") as f:
            lexicon = json.load(f)
        synonyms = None
        if synonyms_path:
            with open(synonyms_path, encoding="utf-8") as f:
                synonyms = json.load(f)
        return cls(frames, lexicon, split, synonyms)

    @property
    def tables(self) -> list[str]:
        """사전에 명사가 있는 테이블 = 이 조합기가 다루는 테이블."""
        return sorted(self.nouns)

    # --- 선택 -------------------------------------------------------------

    def _usable(self, entries: list[Entry]) -> list[Entry]:
        if self.split == "train":
            if self._eval_family is not None:
                own = [e for e in entries if e.split == "eval" and entry_family(e.ref) == self._eval_family]
                if own:
                    return own
            return [e for e in entries if e.split == "train"]
        evals = [e for e in entries if e.split == "eval"]
        return evals or entries

    def _pick(self, entries: list[Entry], rng: random.Random, trace: list[Entry]) -> Entry:
        usable = self._usable(entries)
        if not usable:
            raise ComposeError("쓸 수 있는 표현 없음")
        if self.wildcard_ratio is not None:
            wild = [e for e in usable if is_wildcard(e)]
            if wild and len(wild) < len(usable):
                usable = wild if rng.random() < self.wildcard_ratio else [e for e in usable if not is_wildcard(e)]
        e = rng.choice(usable)
        trace.append(e)
        return e

    def _cond_entries(self, unit: Unit, pos: str) -> list[Entry]:
        """그 조건의 표현. 값 하나짜리 조건은 값 전용 키(예: 'items.stock = 0' → out of stock)의 표현도 함께 쓴다."""
        d = self.conds.get(unit.key)
        if d is None:
            raise ComposeError(f"사전에 없는 조건 {unit.key}")
        if unit.is_list and pos in ("pre", "post"):
            return self._usable(d.get(f"{pos}_or") or d.get(pos, []))
        fixed = self.conds.get(value_key(unit)) if len(unit.values) == 1 else None
        return self._usable(d.get(pos, []) + (fixed.get(pos, []) if fixed else []))

    def _head_entries(self, unit: Unit, bare: bool) -> list[Entry]:
        """명사구 전체가 값인 표현. bare 면 the 로 시작하지 않는 것만 ({np_head} 검색어형 틀용)."""
        heads = self._cond_entries(unit, "head")
        return [e for e in heads if not e.sg.startswith("the ")] if bare else heads

    # --- 명사구 -----------------------------------------------------------

    def noun_phrase(self, rec: dict, kind: str, number: str, noun: Entry, rng: random.Random,
                    trace: list[Entry], tails: list[str]) -> str:
        """kind 의 명사구. LIMIT 없는 정렬 표현은 문장 끝에 붙이도록 tails 에 넣는다
        ("what are the items under $20 called, sorted by price?")."""
        t = rec["table"]
        units = units_of(rec)
        order, limit = rec["order"], rec["limit"]

        # 명사구 전체를 값으로 (item 3001 / the kayak): 조건 하나, 정렬 없음, the 명사구만
        if kind in ("def", "head") and len(units) == 1 and not units[0].is_list and order is None:
            heads = self._head_entries(units[0], bare=kind == "head")
            if heads and (kind == "head" or rng.random() < 0.5):
                return _fill(self._pick(heads, rng, trace).text(number), t, units[0], rng, self._tagger,
                             self.short_words)
        if kind == "head":
            raise ComposeError("명사구 전체를 값으로 쓸 수 없는 SQL")

        # 범주 값이 명사 노릇 (electronics under 50 bucks / sports gear): 값 전용 키의 noun 위치 표현
        noun_text, noun_i = None, None
        is_or = rec["conn"] == "OR"  # 서로 다른 컬럼의 OR: 조건을 모두 명사 뒤에 두고 or 로 잇는다
        if kind != "conds" and not (limit and limit > 1) and not is_or:
            noun_cands = [i for i, u in enumerate(units) if self._cond_entries(u, "noun")]
            if noun_cands and rng.random() < 0.5:
                noun_i = rng.choice(noun_cands)
                noun_num = "sg" if kind == "indef" else "pl" if kind == "any" else number
                noun_text = self._pick(self._cond_entries(units[noun_i], "noun"), rng, trace).text(noun_num)
                units = [u for i, u in enumerate(units) if i != noun_i]

        pre_cands = [i for i, u in enumerate(units) if self._cond_entries(u, "pre")]
        pre_i = (rng.choice(pre_cands) if kind != "conds" and not is_or and pre_cands and rng.random() < 0.5
                 else None)
        pre = ""
        if pre_i is not None:
            pre = _fill(self._pick(self._cond_entries(units[pre_i], "pre"), rng, trace).text(number),
                        t, units[pre_i], rng, self._tagger, self.short_words)
        posts = []
        rest = [u for i, u in enumerate(units) if i != pre_i]
        rng.shuffle(rest)
        for u in rest:
            ents = self._cond_entries(u, "post")
            if not ents:
                raise ComposeError(f"뒤 위치 표현 없음 {u.key}")
            posts.append(_fill(self._pick(ents, rng, trace).text(number), t, u, rng, self._tagger,
                               self.short_words))
        # 조건이 둘 이상이면 연결어를 넣는다 (그냥 이어 붙이면 "names containing 'cu' $30 or more"처럼 경계가 흐려진다)
        if len(posts) >= 2 and is_or:
            # 조건 안에 or / and / 쉼표 / 슬래시가 있으면(값 나열 "in kitchen or sports", "between 5 and 10") 쉼표로 경계를 둔다
            sep = ", or " if any(re.search(r" or | and |,|/", p) for p in posts) else rng.choice([" or ", ", or "])
            post = ", ".join(posts[:-1]) + sep + posts[-1] if len(posts) > 2 else sep.join(posts)
        elif len(posts) >= 2:
            sep = ", " if any(" and " in p or p.startswith("but ") for p in posts) else rng.choice([", ", " and "])
            post = sep.join(posts[:-1]) + (", and " if sep == ", " and len(posts) > 2 else
                                           " and " if sep == " and " else ", ") + posts[-1]
        else:
            post = "".join(posts)

        lead, tail, after_noun = [], "", ""
        if order:
            okey = f"{t}.{order['col']} {order['dir']}"
            od = self.orders.get(okey)
            if od is None:
                raise ComposeError(f"사전에 없는 정렬 {okey}")
            if limit is None:
                tails.append(self._pick(od.get("sort", []), rng, trace).text(number))
            else:
                if kind not in ("def", "nodet"):
                    raise ComposeError("LIMIT 은 the 명사구(또는 the 없는 형태)로만 표현")
                # 조건 표현이 명사 뒤에 있으면 "with the lowest …" 를 붙이지 않는다 ("… with 9 units with the smallest …")
                styles = [s for s in ("sup", "post", "top")
                          if self._usable(od.get(s, [])) and not (s == "top" and limit == 1)
                          and not (s == "post" and posts)]
                # 정렬 표현 + 개수를 문장 끝에 ("sports items, lowest stock first, just 10" / LIMIT 1: "…, just the top one")
                count_ents = self.limit_tails1 if limit == 1 else self.limit_tails
                tail_ok = bool(self._usable(od.get("sort", []))) and bool(self._usable(count_ents))
                if not styles and not tail_ok:
                    raise ComposeError(f"LIMIT 표현 없음 {okey}")
                ratio = LIMIT_TAIL_RATIO / 2 if limit == 1 else LIMIT_TAIL_RATIO  # LIMIT 1 은 최상급 표현이 기본이다
                style = "tail" if tail_ok and (not styles or rng.random() < ratio) else rng.choice(styles)
                n = "" if limit == 1 else str(limit)
                if style == "tail":
                    sort = self._pick(od["sort"], rng, trace).text(number)
                    count = self._pick(count_ents, rng, trace).text(number).replace("{n}", n)
                    tails.append(f"{sort}, {count}")
                    phrase = ""
                else:
                    phrase = self._pick(od[style], rng, trace).text(number)
                # LIMIT 1 은 "the single cheapest item", "the one order with the most units" 로도 쓴다
                if style == "sup":  # the 3 cheapest / the top 3 cheapest / the cheapest 3
                    orders_ = [[n, phrase], ["top", n, phrase]] + ([[phrase, n]] if re.fullmatch(r"[a-z]+", phrase) else [])
                    lead = rng.choice(orders_) if n else ([phrase] if rng.random() < 0.75 else ["single", phrase])
                elif style == "post":
                    lead, tail = ([n] if n else [] if rng.random() < 0.75 else [rng.choice(["one", "single"])]), " " + phrase
                elif style == "top":  # "the top 3 items by price in kitchen" — by 구는 명사 바로 뒤
                    lead, after_noun = ["top", n], phrase

        det = {"def": "the", "bare": "", "indef": "a", "any": "any", "nodet": "", "conds": ""}[kind]
        if kind == "conds":
            return post
        noun_word = noun_text or noun.text("sg" if kind == "indef" else "pl" if kind == "any" else number)
        return " ".join(w for w in [det, *lead, pre, noun_word, after_noun, post] if w) + tail

    # --- 문장 -------------------------------------------------------------

    def frames_for(self, rec: dict) -> tuple[str, str, list[Entry]]:
        """(역할, 수, 쓸 수 있는 틀)."""
        role = role_of(rec)
        number = number_of(rec)
        name = f"attr.{rec['table']}.{rec['select']}" if role == "attr" else role
        if name not in self.frames:
            raise ComposeError(f"틀 없음 {name}")
        if role == "exists" and rec["order"]:
            raise ComposeError("존재 확인 SQL 에 정렬이 붙음 (sql_ast.violations 가 막아야 함)")
        cands = list(self.frames[name])
        if role == "ident" and multi_exists_ok(rec):
            cands += self.frames.get("ident_exists", [])
        # 명사구 뒤에 말이 남는 틀("where do … live?")은 명사구가 짧을 때만 쓴다
        # (조건 여러 개, 값 3개짜리 OR, 정렬이 들어가면 끝의 live / on / exist 가 앞과 너무 멀어진다)
        units = units_of(rec)
        long_np = len(units) >= 2 or rec["order"] is not None or any(len(u.values) >= 3 for u in units if u.is_list)
        ok = []
        for e, num, tables in cands:
            if (num and num != number) or (tables and rec["table"] not in tables):
                continue
            if long_np and frame_has_tail(e.sg):
                continue
            needs_open = any(f"{{np_{k}}}" in e.sg for k in ("bare", "indef", "any", "conds"))
            if needs_open and (rec["limit"] is not None or not rec["conds"]):
                continue
            if "{np_conds}" in e.sg and rec["order"] is not None:
                continue
            if "{np_nodet}" in e.sg and rec["limit"] is None:  # LIMIT 없으면 {np_bare} 틀과 같다
                continue
            if "{np_head}" in e.sg and not (len(units) == 1 and not units[0].is_list and rec["order"] is None
                                            and self._head_entries(units[0], bare=True)):
                continue
            ok.append(e)
        return role, number, ok

    def compose(self, rec: dict, rng: random.Random) -> tuple[str, list[Entry]]:
        """SQL 레코드 → (질문, 쓴 틀·표현 목록)."""
        if rec["table"] not in self.nouns:
            raise ComposeError(f"사전에 없는 테이블 {rec['table']}")
        trace: list[Entry] = []
        role, number, frames = self.frames_for(rec)
        if not self._usable(frames):
            raise ComposeError(f"맞는 틀 없음 {role} {number}")
        frame = self._pick(frames, rng, trace)
        noun = self._pick(self.nouns[rec["table"]], rng, trace)
        # eval 틀에는 유사어를 쓰지 않는다 ("show the prices of" → "display the prices of" 는 학습 틀 "list the prices of"
        # 에 유사어를 쓴 문장과 같아진다)
        text = self.synonyms.apply(frame.sg, rng) if self.synonyms and frame.split == "train" else frame.sg
        tails: list[str] = []
        for kind in NP_KINDS:
            ph = f"{{np_{kind}}}"
            if ph in text:
                text = text.replace(ph, self.noun_phrase(rec, kind, number, noun, rng, trace, tails))
        text = text.replace("{noun_sg}", noun.sg).replace("{noun}", noun.text(number))
        # 명사구로 끝나는 틀이면 정렬 표현을 쉼표 없이도 붙인다 ("list the clothing items sorted by price").
        # 명사구 뒤에 말이 있으면("what are … called") 쉼표를 둔다
        np_last = re.search(r"\{np_\w+\}\??$", frame.sg) is not None
        for tail in tails:
            sep = " " if np_last and NO_COMMA_SORT.match(tail) and rng.random() < TAIL_NO_COMMA_RATIO else ", "
            text = text[:-1] + f"{sep}{tail}?" if text.endswith("?") else f"{text}{sep}{tail}"
        if text.endswith("?") and rng.random() < DROP_QMARK_RATIO:
            text = text[:-1]
        return finish(text), trace

    def eval_families(self, rec: dict) -> list[str]:
        """이 SQL 의 문장에서 eval 표현을 쓸 수 있는 자리 (frame / cond:<키> / order:<키>). 명사는 뺀다."""
        fams = []
        if any(e.split == "eval" for e in self.frames_for(rec)[2]):
            fams.append("frame")
        for u in units_of(rec):
            for key in [u.key] + ([value_key(u)] if len(u.values) == 1 else []):
                if any(e.split == "eval" for rows in self.conds.get(key, {}).values() for e in rows):
                    fams.append(f"cond:{key}")
        if rec["order"]:
            okey = f"{rec['table']}.{rec['order']['col']} {rec['order']['dir']}"
            if any(e.split == "eval" for rows in self.orders.get(okey, {}).values() for e in rows):
                fams.append(f"order:{okey}")
        return fams

    def compose_phrase(self, rec: dict, seed: int, epoch: int, tagged: bool = False, tries: int = 20):
        """phrase 평가용: train 조합에서 eval 자리 하나만 eval 표현으로 바꾼다. 자리는 SQL 마다 무작위로 고르고,
        그 자리가 이 문장에 실제로 나오도록(예: 명사 앞 위치가 뽑혀 eval 표현이 없는 경우) 다른 난수로 다시 조합한다.
        반환: (질문, 태그 SQL 또는 None, 쓴 틀·표현, 자리) / eval 자리가 없거나 끝내 못 넣으면 None."""
        if self.split != "train":
            raise ValueError("compose_phrase 는 train 조합기에서 쓴다")
        fams = self.eval_families(rec)
        random.Random(f"{seed}:{epoch}:{rec['sql']}:family").shuffle(fams)
        for fam in fams:
            self._eval_family = fam
            try:
                for k in range(tries):
                    rng = rng_for(seed, epoch * 1000 + k, rec["sql"])
                    if tagged:
                        q, sql, trace = self.compose_tagged(rec, rng)
                    else:
                        (q, trace), sql = self.compose(rec, rng), None
                    if any(e.split == "eval" for e in trace):
                        return q, sql, trace, fam
            finally:
                self._eval_family = None
        return None

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


def literal_problems(question: str, rec: dict, value_keys: frozenset[str] = frozenset()) -> list[str]:
    """SQL 의 리터럴과 LIMIT n(2 이상)이 질문에 그대로 들어 있는지. 숫자는 값으로 비교한다(20 = 20.00 = $20).
    value_keys(값 전용 사전 키)에 해당하는 조건은 숫자 없는 표현(out of stock)일 수 있어 보지 않는다."""
    nums = {float(x) for x in _NUM_RE.findall(question)}
    out = []
    for c in rec["conds"]:
        if c["op"] != "IN" and f"{rec['table']}.{c['col']} {c['op']} {c['val']}" in value_keys:
            continue
        vals = list(c["val"]) if c["op"] == "IN" else [c["val"]] + ([c["val2"]] if c["op"] == "BETWEEN" else [])
        for v in vals:
            if c["op"] == "LIKE":
                v = v.strip("%")
            if isinstance(v, str):
                if v not in question:
                    out.append(f"값 {v!r} 없음")
            elif float(v) not in nums:
                out.append(f"값 {v} 없음")
    if rec["limit"] and rec["limit"] > 1 and float(rec["limit"]) not in nums:
        out.append(f"LIMIT {rec['limit']} 없음")
    return out


def _value_pos(question: str, val) -> int | None:
    """질문에서 값이 처음 나오는 위치. 텍스트는 단어 경계(뒤의 's/s/es 허용), 숫자는 앞뒤가 숫자·소수점이 아닌 곳."""
    if isinstance(val, str):
        m = re.search(rf"\b{re.escape(val)}(?:'?s|es)?\b", question)
    else:
        m = re.search(rf"(?<![\d.]){val}(?![\d]|\.\d)", question)
    return m.start() if m else None


def question_order_sql(rec: dict, question: str) -> str:
    """라벨 규칙 L8: IN 의 값은 질문에 나온 순서로 쓴다(AND·OR 조건 순서는 정규형 그대로). rec 은 정규형 SQL 레코드.
    값을 질문에서 찾지 못하면 ComposeError."""
    q = from_dict(rec)
    if not any(c.op == "IN" for c in q.conds):
        return rec["sql"]
    conds = []
    for c in q.conds:
        if c.op == "IN":
            pos = [_value_pos(question, v) for v in c.val]
            if None in pos or len(set(pos)) != len(pos):
                raise ComposeError(f"IN 값 위치를 찾지 못함: {question!r} <- {rec['sql']!r}")
            c = Cond(c.col, "IN", tuple(v for _, v in sorted(zip(pos, c.val), key=lambda x: x[0])))
        conds.append(c)
    return Query(q.table, q.select, tuple(conds), q.conn, q.order, q.limit).render()


def value_keys(lexicon: dict) -> frozenset[str]:
    return frozenset(k for k in lexicon["conds"] if is_value_key(k))


def lexicon_conflicts(lexicon: dict) -> list[str]:
    """같은 테이블·같은 위치에 같은 표현이 서로 다른 SQL 조각 키로 들어 있으면 라벨이 갈린다.
    위치가 다르면(명사 앞 'kitchen items' 와 명사구 전체 'kayak') 문장 모양이 달라 충돌이 아니고,
    테이블이 다르면('customers in boston' 과 'items in kitchen') 명사가 테이블을 정해 충돌이 아니다."""
    family = {"pre_or": "pre", "post_or": "post"}

    def value_decides(keys: set[str]) -> bool:
        """범주형 컬럼끼리이고 값 목록이 서로 겹치지 않으면 값이 컬럼을 정한다 ('gold customers' / 'boston customers')."""
        domains = []
        for k in keys:
            section, frag = k.split(":", 1)
            table_col = tuple(frag.split(" ", 1)[0].split("."))
            if section != "conds" or table_col not in CAT_VALUES:
                return False
            domains.append(set(CAT_VALUES[table_col]))
        return sum(len(d) for d in domains) == len(set().union(*domains))

    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for section in ("conds", "orders"):
        for key, d in lexicon[section].items():
            table = key.split(".", 1)[0]
            for pos, rows in d.items():
                for r in rows:
                    e = _entry(r, "")
                    for s in {e.sg, e.pl}:
                        seen[(table, family.get(pos, pos), s)].add(f"{section}:{key}")
    return [f"{t} {pos} {s!r}: {sorted(keys)}" for (t, pos, s), keys in sorted(seen.items())
            if len(keys) > 1 and not value_decides(keys)]


def check(records: dict[str, list[dict]], frames: dict, lexicon: dict, epochs: int = 3, seed: int = 0,
          synonyms: dict | None = None) -> dict:
    """records: {이름: SQL 레코드}. 사전이 다루는 테이블만 본다. 실패 목록과 통계를 돌려준다.
    이름별 조합기는 COMPOSER_SPLIT 을 따르고(없으면 train), PHRASE_SOURCE 가 있으면 phrase 평가도 함께 본다."""
    fails: list[str] = [f"사전 충돌 {c}" for c in lexicon_conflicts(lexicon)]
    if synonyms:
        fails += synonym_problems(synonyms, frames, lexicon)
    vkeys = value_keys(lexicon)
    stats: dict = {}
    jobs = [(name, rows, COMPOSER_SPLIT.get(name, "train")) for name, rows in records.items()]
    if PHRASE_SOURCE in records:
        jobs.append(("phrase", records[PHRASE_SOURCE], "phrase"))
    for name, rows, split in jobs:
        comp = Composer(frames, lexicon, "train" if split == "phrase" else split, synonyms)
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
        # 조합기 문장은 라벨 규칙대로 만든다. 판정(clean_ambiguous)과 어긋나는 문장이 0.1% 를 넘으면 틀·표현과 규칙이 갈린 것이다
        if rule_mismatch > 0.001 * max(1, sum(distinct)):
            fails.append(f"[{name}] 라벨 규칙 판정과 어긋난 문장 {rule_mismatch}개 (clean_ambiguous.drop_reason)")
    return {"fails": fails, "stats": stats}


# ---------------------------------------------------------------------------
# 명령행
# ---------------------------------------------------------------------------

def _load_records(paths: list[str]) -> dict[str, list[dict]]:
    out = {}
    for p in paths:
        name = Path(p).stem
        with open(p, encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def _sample(rows: list[dict], comp: Composer, n: int, seed: int, epoch: int) -> list[dict]:
    rows = [r for r in rows if r["table"] in comp.tables]
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    out = []
    for r in picked:
        q, trace = comp.compose(r, rng_for(seed, epoch, r["sql"]))
        out.append({"question": q, "sql": r["sql"], "frame": trace[0].ref})
    return out


def build(records: dict[str, list[dict]], comp: Composer, epochs: int, seed: int, out: Path,
          stage1_pairs: str) -> int:
    """학습 쌍 파일: SQL 마다 epoch 0..epochs-1 의 문장 (train 조합기). 1단계 학습 쌍은 같은 폴더에 사본으로 둔다
    (학습 폴더의 *.json 전부가 학습 데이터·BPE 코퍼스다). 모델 최대 길이를 넘는 쌍은 train.py 가 뺀다.
    sql 은 IN 값을 질문에 나온 순서로 쓴 표기(L8)이고, sql_group 은 정규형 SQL 이다(train.py 가 검증셋을 이것으로 나눈다)."""
    pairs = []
    for rows in records.values():
        for r in rows:
            for ep in range(epochs):
                q = comp.compose(r, rng_for(seed, ep, r["sql"]))[0]
                pairs.append({"question": q, "sql": question_order_sql(r, q), "sql_group": r["sql"],
                              "source": "composer"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(f"{len(pairs)}개 -> {out}")
    if stage1_pairs:
        # 1단계 쌍은 1단계 규칙으로 정제됐다. 2단계 규칙(예: L13 의 find = star)과 어긋나는 쌍은 사본에서 뺀다
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
              f"(2단계 라벨 규칙과 어긋난 {len(removed)}개 제외 -> {PATHS.raw_dir / 'stage1_pairs_removed.json'})")
    return 0


def main() -> int:
    sql_dir = PATHS.sql_dir
    ap = argparse.ArgumentParser(description="2단계 질문 조합기")
    ap.add_argument("command", choices=["sample", "check", "dump", "build"])
    ap.add_argument("--sql", nargs="+", default=None,
                    help=f"SQL 레코드 파일 (기본: sample·dump 는 {sql_dir}/train.json, check 는 {sql_dir} 의 4개 분할)")
    ap.add_argument("--frames", default=str(FRAMES_PATH), help=f"문장 틀 (기본 {FRAMES_PATH})")
    ap.add_argument("--lexicon", default=str(LEXICON_PATH), help=f"표현 사전 (기본 {LEXICON_PATH})")
    ap.add_argument("--synonyms", default=str(SYNONYMS_PATH), help=f"유사어 사전, '' 이면 쓰지 않음 (기본 {SYNONYMS_PATH})")
    ap.add_argument("--split", choices=SPLITS, default="train", help="sample·dump: train 표현만 / eval 표현 우선")
    ap.add_argument("--n", type=int, default=20, help="sample·dump: SQL 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0, help="sample·dump: 같은 SQL 의 다른 문장을 보려면 바꾼다")
    ap.add_argument("--epochs", type=int, default=3, help="check: SQL 마다 조합해 볼 epoch 수 / build: SQL 당 문장 수")
    ap.add_argument("--out", default=None, help="dump: 저장할 JSON 파일 (예: data_raw/stage2/compose_pilot.json) / "
                                                f"build: 학습 쌍 파일 (기본 {PATHS.train_dir}/composed.json)")
    ap.add_argument("--stage1-pairs", default=str(stage_paths(1).train_dir / "pairs.json"),
                    help="build: --out 옆에 stage1_pairs.json 으로 복사할 1단계 학습 쌍 ('' 이면 복사 안 함)")
    ap.add_argument("--wildcard-ratio", type=float, default=WILDCARD_RATIO,
                    help=f"sample·dump·build: LIKE 표현 중 와일드카드 표현을 고를 확률 (기본 {WILDCARD_RATIO})")
    args = ap.parse_args()

    with open(args.frames, encoding="utf-8") as f:
        frames = json.load(f)
    with open(args.lexicon, encoding="utf-8") as f:
        lexicon = json.load(f)
    synonyms = None
    if args.synonyms:
        with open(args.synonyms, encoding="utf-8") as f:
            synonyms = json.load(f)

    if args.command == "check":
        paths = args.sql or [str(sql_dir / n) for n in ("train.json", "eval_indist.json", "eval_struct.json",
                                                         "eval_holdout.json")]
        result = check(_load_records(paths), frames, lexicon, args.epochs, args.seed, synonyms)
        for name, st in result["stats"].items():
            print(f"[{name}] SQL {st['n_sql']}개, 실패 {st['n_failed']}, SQL 당 서로 다른 문장 "
                  f"{st['avg_distinct_per_sql']} / {args.epochs}, 쓴 틀 {st['frames_used']}종, 라벨 규칙 판정과 어긋남 {st['n_rule_mismatch']}")
        fails = result["fails"]
        print("통과" if not fails else f"실패 {len(fails)}건\n  - " + "\n  - ".join(fails[:30]))
        return 1 if fails else 0

    if args.command == "build":
        return build(_load_records(args.sql or [str(sql_dir / "train.json")]),
                     Composer(frames, lexicon, "train", synonyms, args.wildcard_ratio),
                     args.epochs, args.seed, Path(args.out or PATHS.train_dir / "composed.json"), args.stage1_pairs)

    comp = Composer(frames, lexicon, args.split, synonyms, args.wildcard_ratio)
    rows = [r for rs in _load_records(args.sql or [str(sql_dir / "train.json")]).values() for r in rs]
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
