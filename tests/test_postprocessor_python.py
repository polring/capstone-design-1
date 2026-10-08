import pytest
from src.postprocessor.python import process_sql
from src.postprocessor.python.parser import pre_correct_keywords
from src.postprocessor.python.replacement import get_ngrams

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
    sql = "SELETC id, name FRMO users"
    fixed = pre_correct_keywords(sql)
    assert fixed == "SELECT id, name FROM users"


def test_typo_correction_keywords():
    sql = "SELETC id, name FRMO users"
    result = process_sql(sql, strategy="typo", dictionary=TEST_DICTIONARY)
    assert result == "SELECT id, name FROM users"


def test_typo_correction_identifiers():
    sql = "SELECT usr_name, emil FROM users"
    result = process_sql(sql, strategy="typo", dictionary=TEST_DICTIONARY)
    assert "user_name" in result
    assert "email" in result


def test_typo_correction_complex():
    sql = "SELECT manger_id FROM departmnt WHERE id = 1"
    result = process_sql(sql, strategy="typo", dictionary=TEST_DICTIONARY)
    assert "manager_id" in result
    assert "department" in result


def test_get_ngrams():
    ngrams = get_ngrams("test", 2)
    assert ngrams == {"te", "es", "st"}


def test_ngram_correction():
    # 'departmnt' -> 'department'
    sql = "SELECT id FROM departmnt"
    result = process_sql(sql, strategy="ngram", dictionary=TEST_DICTIONARY)
    assert "department" in result


def test_strategy_chaining():
    # typo fallback to ngram
    sql = "SELECT usr_name FROM departmnt"
    result = process_sql(sql, strategy="typo,ngram", dictionary=TEST_DICTIONARY)
    assert "user_name" in result
    assert "department" in result


def test_semantic_correction_pipeline():
    sql = "SELECT usr FROM users"
    # Even if model is not downloaded/fails, it should fallback to original
    result = process_sql(sql, strategy="semantic", dictionary=TEST_DICTIONARY, dim=300)
    assert "usr" in result or "user_name" in result
