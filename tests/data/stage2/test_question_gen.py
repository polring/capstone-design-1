import json

import pytest

from src.data.stage2 import question_gen as qg
from src.data.stage2.sql_ast import Cond, Query, canonical, to_dict

pytestmark = pytest.mark.unit


def rec(table, select, conds=(), conn=None, order=None, limit=None, **meta) -> dict:
    return dict(to_dict(canonical(Query(table, select, tuple(conds), conn, order, limit))), **meta)


IN_REC = rec("customers", "*", [Cond("city", "IN", ("boston", "denver")), Cond("membership", "=", "gold")], "AND")
LIKE_REC = rec("items", "item_name", [Cond("item_name", "LIKE", "jo%"), Cond("price", "<", 20.0)], "AND",
               ("price", "ASC"), 3)


def test_prompt_has_sql_personas_and_no_composer_text():
    """프롬프트에는 SQL 과 페르소나가 들어가고, 조합기 표현(예: 정렬 뒤 개수 'just {n}')은 들어가지 않는다"""
    p = qg.build_batch_prompt([(7, IN_REC), (9, LIKE_REC)])
    assert f"- id=7 | {IN_REC['sql']}" in p and f"- id=9 | {LIKE_REC['sql']}" in p
    assert all(name in p for name in qg.PERSONAS)
    assert "just {n}" not in p and "sorted by" not in p


def test_literal_and_label_checks():
    """값·LIKE 조각·LIMIT 이 다 있어야 통과하고, 라벨은 IN 값을 질문 순서로 쓴다. 대소문자는 보지 않는다"""
    assert qg.check_question("Gold members in Denver or Boston?", IN_REC) == []
    assert any("boston" in f for f in qg.check_question("gold members in denver", IN_REC))
    assert qg.check_question("the 3 cheapest items starting with jo under $20", LIKE_REC) == []
    assert any("LIMIT" in f for f in qg.check_question("cheapest items starting with jo under $20", LIKE_REC))
    assert qg.check_question("the three cheapest items starting with jo under $20 3", LIKE_REC) == []  # 숫자 단어는 flag 만
    assert any("숫자" in f for f in qg.soft_flags("the three cheapest items starting with jo under $20 3", LIKE_REC))
    out_of_stock = rec("items", "*", [Cond("stock", "=", 0)])
    assert qg.check_question("items that are out of stock", out_of_stock) == []  # L16 고정값
    assert any("밑줄" in f for f in qg.check_question("item_name of the 3 cheapest starting with jo under 20", LIKE_REC))


def test_validate_round(tmp_path):
    """플랫폼 폴더마다 쌍·통과율을 내고, 같은 질문이 다른 SQL 에 붙으면 양쪽 다 뺀다"""
    # SELECT 만 다른 SQL: 값 검사는 둘 다 통과하는 문장이 생긴다
    other = rec("customers", "membership", [Cond("city", "IN", ("boston", "denver")), Cond("membership", "=", "gold")], "AND")
    meta = {1: dict(IN_REC, split="indist", batch=1), 2: dict(other, split="indist", batch=1)}
    (tmp_path / qg.ROUND_SQL).write_text(json.dumps(meta), encoding="utf-8")
    good = ["gold denver boston members", "full records of gold ppl in boston or denver", "members gold in denver and boston",
            "could you list the gold members living in boston or denver for our mailing", "need gold members, denver/boston"]
    resp = [{"id": 1, "questions": good},
            {"id": 2, "questions": ["names gold boston denver", "full records of gold ppl in boston or denver", "x", "y", "z"]}]
    (tmp_path / "responses" / "gpt").mkdir(parents=True)
    (tmp_path / "responses" / "gpt" / "batch_01.json").write_text(json.dumps(resp), encoding="utf-8")
    report = qg.validate(str(tmp_path))
    pairs = json.loads((tmp_path / "pairs_gpt.json").read_text(encoding="utf-8"))
    qs = {p["question"] for p in pairs}
    assert "full records of gold ppl in boston or denver" not in qs  # 다른 SQL 과 같은 질문
    assert "names gold boston denver" in qs and "x" not in qs
    p = next(p for p in pairs if p["question"] == "gold denver boston members")
    assert p["sql"] == ("SELECT * FROM customers WHERE city IN ('denver', 'boston') AND membership = 'gold'")
    assert p["sql_group"] == IN_REC["sql"] and p["persona"] == "keyword" and p["platform"] == "gpt"
    assert report["platforms"]["gpt"]["pass_rate"]["keyword"] == 1.0
