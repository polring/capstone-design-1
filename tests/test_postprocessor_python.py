import pytest
from src.postprocessor.python import process_sql
from src.postprocessor.python.parser import pre_correct_keywords
from src.postprocessor.python.replacement import get_ngrams

# 테스트에 사용할 가상의 데이터베이스 스키마(테이블 및 컬럼명) 사전
TEST_DICTIONARY = [
    "users",
    "user_name",
    "email",
    "id",
    "name",
    "department",
    "manager_id",
]


def test_pre_correct_keywords():
    """
    파서(sqlglot)로 넘어가기 전, 문자열 상태에서 SQL 예약어 오타가
    정상적으로 사전 교정(Pre-correction)되는지 검증합니다.
    예: SELETC -> SELECT, FRMO -> FROM
    """
    sql = "SELETC id, name FRMO users"
    fixed = pre_correct_keywords(sql)
    assert fixed == "SELECT id, name FROM users"


def test_typo_correction_keywords():
    """
    전체 파이프라인(process_sql)을 돌렸을 때, 예약어 오타가
    포함되어 있어도 파서 에러 없이 정상적으로 최종 교정되는지 검증합니다.
    """
    sql = "SELETC id, name FRMO users"
    result = process_sql(sql, strategy="typo", dictionary=TEST_DICTIONARY)
    assert result == "SELECT id, name FROM users"


def test_typo_correction_identifiers():
    """
    Typo(편집 거리 기반) 전략을 사용하여, 단어의 길이가 비슷하고
    단순 오타가 난 식별자(컬럼/테이블명)가 알맞게 교정되는지 검증합니다.
    예: usr_name -> user_name, emil -> email
    """
    sql = "SELECT usr_name, emil FROM users"
    result = process_sql(sql, strategy="typo", dictionary=TEST_DICTIONARY)
    assert "user_name" in result
    assert "email" in result


def test_typo_correction_complex():
    """
    여러 개의 식별자 오타가 한 쿼리에 복합적으로 존재할 때,
    모든 오타가 누락 없이 전부 교정되는지 검증합니다.
    """
    sql = "SELECT manger_id FROM departmnt WHERE id = 1"
    result = process_sql(sql, strategy="typo", dictionary=TEST_DICTIONARY)
    assert "manager_id" in result
    assert "department" in result


def test_get_ngrams():
    """
    N-gram 분리 유틸리티 함수가 단어를 지정된 길이(기본 2)만큼
    정확히 분리하여 집합(Set)으로 반환하는지 검증합니다.
    """
    ngrams = get_ngrams("test", 2)
    assert ngrams == {"te", "es", "st"}


def test_ngram_correction():
    """
    N-gram(부분 문자열 Jaccard 유사도) 전략이 제대로 동작하는지 검증합니다.
    길이가 긴 단어에서 일부 글자가 누락되었을 때(departmnt -> department) 유용합니다.
    """
    sql = "SELECT id FROM departmnt"
    result = process_sql(sql, strategy="ngram", dictionary=TEST_DICTIONARY)
    assert "department" in result


def test_strategy_chaining():
    """
    전략 체이닝(다중 적용) 기능을 검증합니다.
    'typo,ngram'으로 지정했을 때 Typo 전략으로 처리하지 못한 것을
    N-gram이 Fallback으로 받아서 처리하는지 확인합니다.
    """
    sql = "SELECT usr_name FROM departmnt"
    result = process_sql(sql, strategy="typo,ngram", dictionary=TEST_DICTIONARY)
    assert "user_name" in result
    assert "department" in result


def test_semantic_correction_pipeline():
    """
    Semantic(의미론적 유사도) 전략 파이프라인의 안전성을 검증합니다.
    오프라인 캐시나 GloVe 모델 로드에 실패하더라도 파이프라인이
    죽지 않고 원본 단어(usr)나 유사 단어(user_name)로 에러 없이 반환되는지 확인합니다.
    """
    sql = "SELECT usr FROM users"
    result = process_sql(sql, strategy="semantic", dictionary=TEST_DICTIONARY, dim=300)
    assert "usr" in result or "user_name" in result
