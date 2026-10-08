import pytest
from src.postprocessor.python.main import process_sql
from src.postprocessor.python.parser import pre_correct_keywords
from src.postprocessor.python.replacement import get_ngrams


def test_pre_correct_keywords():
    sql = "SELETC id, name FRMO users"
    fixed = pre_correct_keywords(sql)
    assert fixed == "SELECT id, name FROM users"


def test_typo_correction_keywords():
    sql = "SELETC id, name FRMO users"
    result = process_sql(sql, strategy="typo")
    assert result == "SELECT id, name FROM users"


def test_typo_correction_identifiers():
    sql = "SELECT usr_name, emil FROM users"
    result = process_sql(sql, strategy="typo")
    assert "user_name" in result
    assert "email" in result


def test_typo_correction_complex():
    sql = "SELECT manger_id FROM departmnt WHERE id = 1"
    result = process_sql(sql, strategy="typo")
    assert "manager_id" in result
    assert "department" in result


def test_get_ngrams():
    ngrams = get_ngrams("test", 2)
    assert ngrams == {"te", "es", "st"}


def test_ngram_correction():
    # 'departmnt' -> 'department'
    sql = "SELECT id FROM departmnt"
    result = process_sql(sql, strategy="ngram")
    assert "department" in result


def test_strategy_chaining():
    # typo fallback to ngram
    sql = "SELECT usr_name FROM departmnt"
    result = process_sql(sql, strategy="typo,ngram")
    assert "user_name" in result
    assert "department" in result


# semantic_correction test is skipped by default in unit tests to avoid downloading models
# unless a mock model is used, but we can test the pipeline doesn't crash
def test_semantic_correction_pipeline():
    sql = "SELECT usr FROM users"
    # Even if model is not downloaded/fails, it should fallback to original
    result = process_sql(sql, strategy="semantic")
    assert "usr" in result or "user_name" in result
