"""
sql_rules.py — 1단계 SQL 형식 규칙 (SELECT 컬럼 하나 또는 *, FROM 테이블 하나, WHERE 등호 하나)

evaluation.py 가 평가 단계(config.STAGE 또는 체크포인트의 stage)에 맞는 src.data.stage<N>.sql_rules 를 불러
단계마다 다른 채점 규칙을 적용한다. 단계마다 같은 이름으로 아래를 제공한다:

    parse_sql(sql) -> dict | None              이 단계 문법으로 분해 (아니면 None)
    error_parts(gold, pred) -> list[str]       오답에서 틀린 부분 이름 목록
    select_ambiguous(question, gold) -> bool   질문만으로 정답이 하나로 정해지지 않아 다중 정답을 인정하는 문항인가
    multi_match(question, gold, pred) -> bool  다중 정답 EM 기준으로 맞았는가
    STAGE, MULTI_NOTE                          단계 번호, 리포트에 쓰는 다중 정답 설명

clean_ambiguous.py(라벨 규칙 정제)도 같은 규칙을 쓴다.
"""

from __future__ import annotations

import re

STAGE = 1
SQL_PATTERN = re.compile(r"^SELECT (\*|\w+) FROM (\w+) WHERE (\w+) = (.+)$")
SQL_PARTS = ("table", "where_col", "select", "literal")
MULTI_NOTE = "SELECT 모호 문항: * ↔ 식별 컬럼도 정답 인정"


def parse_sql(sql: str | None) -> dict | None:
    """1단계 문법(SELECT x FROM t WHERE c = v)으로 분해한다. 맞지 않으면 None."""
    m = SQL_PATTERN.match(sql or "")
    if not m:
        return None
    return dict(zip(("select", "table", "where_col", "literal"), m.groups()))


def error_parts(gold: str, pred: str | None) -> list[str]:
    """정답과 다른 부분 목록. 1단계 문법으로 분해되지 않으면 ['malformed']."""
    g, p = parse_sql(gold), parse_sql(pred)
    if g is None or p is None:
        return ["malformed"]
    return [k for k in SQL_PARTS if g[k] != p[k]]


# 다중 정답 EM: "which item costs 14.46?" 처럼 개체를 묻지만 이름(식별 컬럼)만 원하는지 행 전체(*)를
# 원하는지 질문에 드러나지 않는 문항은 둘 중 무엇을 SELECT 해도 정답으로 본다. 판정은 질문과 정답 SQL 만
# 보고 하며 모델 출력은 보지 않는다. 기존 EM 은 그대로 두고 별도 지표(em_multi)로 보고한다.
ID_COL = {"customers": "name", "items": "item_name", "orders": "order_id"}
STAR_CUE = re.compile(
    r"\b(all|everything|every|full|whole|complete|entire|details?|info|information|records?|entry|entries"
    r"|rows?|data|profile|lowdown|rundown|about|overview|summary|anything|columns?|fields?"
    r"|on file|scoop|deal with|going on)\b")
ID_CUE = {
    "customers": re.compile(r"\b(names?|named|called)\b"),
    "items": re.compile(r"\b(names?|named|called)\b"),
    "orders": re.compile(r"\b(order[ _]?ids?|order numbers?|ids?|numbers?)\b"),
}


def select_ambiguous(question: str, gold: str) -> bool:
    """정답 SELECT 가 * 또는 식별 컬럼이고, 질문에 둘 중 어느 쪽을 원하는지 단서가 없으면 True.
    SELECT 와 WHERE 컬럼이 같은 존재 확인형(규칙 10)은 표기 규칙이 정해져 있으므로 제외한다."""
    g = parse_sql(gold)
    if g is None:
        return False
    id_col = ID_COL[g["table"]]
    if g["select"] not in ("*", id_col) or g["select"] == g["where_col"]:
        return False
    q = question.lower()
    # WHERE 조건을 말하는 표현("item id 3155", "named vivian")은 SELECT 단서가 아니다
    if g["table"] == "orders":
        q = re.sub(r"\b(item|customer)[ _]?ids?\b", " ", q)
    q = re.sub(rf"\b(named|called)\s+{re.escape(g['literal'].strip(chr(39)))}\b", " ", q)
    return not STAR_CUE.search(q) and not ID_CUE[g["table"]].search(q)


def multi_match(question: str, gold: str, pred: str | None) -> bool:
    """다중 정답 EM: 완전 일치이거나, 모호한 문항에서 SELECT 만 * ↔ 식별 컬럼으로 다른 경우."""
    if pred == gold:
        return True
    if not select_ambiguous(question, gold):
        return False
    g, p = parse_sql(gold), parse_sql(pred)
    return (p is not None and p["select"] in ("*", ID_COL[g["table"]])
            and all(g[k] == p[k] for k in ("table", "where_col", "literal")))
