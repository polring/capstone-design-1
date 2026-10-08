import pytest
from src.postprocessor.python.main import process_sql
from src.postprocessor.python.parser import pre_correct_keywords


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
