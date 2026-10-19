# 역할: 2단계 질문과 SQL의 라벨 규칙 위반·모호성을 검사한다.
# 사용처: 2단계 학습 데이터 정제와 평가의 규칙 일치 여부 판정에 사용한다.

"""2단계 라벨 규칙 판정 (1단계 clean_ambiguous 의 2단계판): 질문과 정답 SQL 만 보고 정답이 라벨 규칙과 어긋나는지 정한다.

규칙은 1단계 L1~L4(2단계에도 그대로 적용)와 2단계 L13~L15(docs/text-to-sql-label-rules.md)다. 여기서는 그중 질문의 단서로
판정할 수 있는 것을 적용한다.
  L1  존재 확인    — SELECT 가 = 조건 컬럼(조건 하나)인 SQL 은 존재 확인 질문에만 쓴다
  L2  * / 식별 컬럼 — 단서가 없으면 1단계 select_label 순서로 정한다 (조회 동사에 L13 star 단서 list·find 와 그 유사어를 더함)
  L3  자기 ID      — item_id(items)·customer_id(customers)는 질문이 ID 를 요구할 때만 SELECT 한다
  L4  테이블       — customers/items 로도 성립하는 orders SQL 은 주문 단서가 있을 때만 쓴다 (L15)
  L14 존재 확인    — 존재 확인 질문(조건 여러 개, 또는 = 가 아닌 조건)은 식별 컬럼을 SELECT 한다. `*` 정답만 판정한다

속성 SELECT(L13 "속성 명사가 있으면 그 컬럼")는 판정하지 않는다. 속성을 컬럼 이름 없이 묻는 말("where does order 5017 stand",
"what's a stopwatch going for")이 학습 문장에도 많아, 단서 목록으로 판정하면 맞는 문항을 버린다(버리면 점수가 부푼다).

단서 목록은 라벨 규칙 문서와 학습 문장(조합기 틀·유사어 사전·명사 사전, 1단계 학습 쌍)에서 정한다. 평가 오답을 보고 고치지
않는다(CLAUDE.md "Rule-consistent EM"). 학습 쌍에 돌려 거의 걸리지 않는지로 확인한다(라벨 규칙 문서 "2단계 판정").
WHERE 조건을 말하는 표현("customer 1102", "named amy", orders 질문의 "item id")은 SELECT 단서로 보지 않는다.
이 판정은 외부 질문(평가셋)에 쓴다: question_gen validate 가 어긋난 문항을 빼고 출처별로 버린 수를 기록한다.
"""

from __future__ import annotations

import re

from src.data.stage1.clean_ambiguous import ORDER_CUE, PLURAL, ROW_VERB
from src.data.stage1.sql_rules import ID_COL, ID_CUE, STAR_CUE
from src.data.stage2 import schema, sql_ast

REASONS = {"L1": "L1 존재 확인", "L2": "L2 * / 식별 컬럼", "L3": "L3 자기 ID", "L4": "L4 테이블", "L14": "L14 존재 확인"}

# L13 star 단서: show, list, find, pull up, look up … 과 조합기 유사어 사전의 대체어(display, let me see, dig up, bring up,
# retrieve, get, enumerate, give me, send me, provide, fetch, grab)
ROW_VERB2 = re.compile(ROW_VERB.pattern + r"|\b(list|find|display|let me see|dig up|bring up|retrieve|enumerate"
                       r"|give me|get me|send me|provide|fetch|grab|get|fill me in)\b")
# 문장 첫머리의 감탄사·접속사 ("hey, who're …", "um, which …", "so is there …")
INTERJ = r"((and|so|ok|okay|hey|um|uh|well|hmm)\W+)?"
# L2-② 의문사 which / who 의문문: 문장 첫머리(감탄사 뒤 포함), 대시 뒤("item 3325 - which orders?"), 쉼표 뒤 which("clothing, which
# items …"), 묻는 동사 뒤("look up which order …"). "shoppers who live in …", "…, who live in denver" 의 관계대명사는 의문문이 아니다
WH_ID2 = re.compile(r"(^|[;:?–—-])\W*" + INTERJ + r"(which|who)\b|,\s*which\b"
                    r"|\b(look up|find out|find|list|show( me)?|tell me|let me know|know|see|check|confirm|figure out"
                    r"|wondering)\s+(which|who)\b")
# L2-③ 복수 명사: 1단계 목록 + 조합기 명사 사전의 복수형 + 범주 값이 명사 노릇을 하는 표현(조합기 사전의 noun 위치) 중
# 값 이름과 다른 것. clothing, electronics 처럼 값 이름 그대로인 말은 조건("in electronics")일 수 있어 넣지 않는다
PLURAL2 = re.compile(PLURAL.pattern + r"|\b(articles|things|clients|members|patrons|shoppers|purchases|transactions"
                     r"|clothes|kitchen stuff|kitchenware|sports equipment|sports gear)\b")
OWN_ID_CUE = re.compile(r"\b(ids?|identifiers?|numbers?)\b|#")
# L1 단서: 1단계(is there, exist, confirm, verify, check) + L14(are there, do we have, do we carry, any … at all).
# 활용형(confirming, checking), there's, 유사어(do you have, offer), 부사가 낀 "do we even / really have"도 같은 뜻이다
# (1단계 학습 쌍·유사어 사전, LLM 학습 질문 llm_r2)
DO_HAVE = r"(do|does) (we|you) ((even|really|actually|still) )?(have|carry|offer)"
EXISTS_CUE = re.compile(r"\b(is there|are there|there'?s|there (is|are)|exists?|existing|confirm\w*|verif\w*|check\w*"
                        r"|" + DO_HAVE + r"|we have (an?|any))\b|\bany\b.*\bat all\b")
# L14: 질문 첫머리의 존재 확인. "what details exist for …", "is there any stock left …"(속성) 같은 문장 속 표현은 보지 않는다
EXISTS_START = re.compile(r"^\W*" + INTERJ + r"((can|could|would) you |please )?"
                          r"(is there|are there|" + DO_HAVE + r"|do any\b.*\bexist|(you )?got any"
                          r"|(confirm|verify|check)( if| whether)?( we have| there (is|are)))\b")


def _strip_conditions(q: str, g: sql_ast.Query, *, schema_module=schema) -> str:
    """WHERE 조건을 말하는 표현을 지운다: ID 조건 값 앞의 id / number / #, orders 질문의 item id / customer id,
    이름 조건의 named / called. ID 조건이 아닌 숫자 앞의 "item id"("item id 174.13" = price 조건)는 SELECT 단서로 남긴다."""
    ids = [str(v) for c in g.conds if schema_module.kind(g.table, c.col) == "id"
           for v in (c.val if c.op == "IN" else (c.val,))]
    if ids:
        alt = "|".join(ids)
        q = re.sub(rf"\b(ids?|numbers?|no\.?)\s*#?\s*(?=({alt})\b)", " ", q)
        q = re.sub(rf"#\s*(?=({alt})\b)", " ", q)
    if g.table == "orders":
        q = re.sub(r"\b(item|customer|product|client)[ _]?(ids?|numbers?)\b", " ", q)
    if any(c.col == ID_COL[g.table] and c.op in ("=", "IN") for c in g.conds):
        q = re.sub(r"\b(named|called|go(es)? by|with the names?|names? of)\b", " ", q)
    return q


def _is_l1(g: sql_ast.Query) -> bool:
    return len(g.conds) == 1 and g.conds[0].op == "=" and g.select == g.conds[0].col


def select_label(q: str, g: sql_ast.Query) -> str:
    """L2: 단서가 없을 때 정답 SELECT (* 또는 식별 컬럼). 1단계 select_label 과 같은 순서다."""
    ident = ID_COL[g.table]
    if any(c.col == ident and c.op in ("=", "IN") for c in g.conds):  # ① WHERE 가 이미 식별 컬럼
        return "*"
    if WH_ID2.search(q):       # ② which / who 의문문
        return ident
    if PLURAL2.search(q):      # ③ 복수형 목록
        return "*"
    if ROW_VERB2.search(q):    # ④ 조회 동사
        return "*"
    return ident               # ⑤ "what item costs 91.52?", "item priced 199.11?"


def drop_reason(pair: dict, *, stage: int = 2) -> str | None:
    """정답이 라벨 규칙과 어긋나면 사유 키(REASONS), 아니면 None. 2단계 문법 밖 SQL 은 판정하지 않는다."""
    try:
        if stage >= 3:
            from src.data.stage3 import schema as active_schema, sql_ast as active_ast
        else:
            active_schema, active_ast = schema, sql_ast
        g = active_ast.parse(pair["sql"])
        if hasattr(g, "aggregate"):
            return None
    except ValueError:
        return None
    q = pair["question"].lower().strip()
    if stage >= 3:
        q = q.replace("_", " ")
    t, sel = g.table, g.select
    if _is_l1(g):
        if stage >= 3:
            intent = q.split(";", 1)[1] if re.match(r"^i(?: am|'m) checking\b", q) and ";" in q else q
            indirect = re.search(r"^does (?:any|an?|the) (?:item|product|customer|order|record)\b.*\bhave\b"
                                 r"|^does\b.*\bappear in\b.*\b(?:orders|records|database|table)\b"
                                 r"|^is\b.*\bpresent\b", intent.strip())
            return None if EXISTS_CUE.search(intent) or indirect else "L1"
        return None if EXISTS_CUE.search(q) else "L1"
    ident = ID_COL[t]
    qs = _strip_conditions(q, g, schema_module=active_schema)
    # 존재 확인 질문이면 L14 가 SELECT 를 정한다. "is there anything on order 5432?"처럼 행 전체 단서가 있으면 조회(*)다
    if g.conds and g.order is None and EXISTS_START.search(q) and not STAR_CUE.search(qs):
        if sel == ident:
            return None
        if sel == "*":
            return "L14"
    if sel in ("*", ident) and not STAR_CUE.search(qs) and not ID_CUE[t].search(qs) and sel != select_label(qs, g):
        return "L2"
    if sel == {"items": "item_id", "customers": "customer_id"}.get(t) and not OWN_ID_CUE.search(qs):
        return "L3"
    cond_cols = {c.col for c in g.conds}
    if (t == "orders" and cond_cols and cond_cols <= {"customer_id", "item_id"} and sel in {"*"} | cond_cols
            and not ORDER_CUE.search(q)):
        return "L4"
    return None
