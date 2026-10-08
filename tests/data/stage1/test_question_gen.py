import json

from src.data.stage1 import question_gen as qg


def _entry(table, where_col, where_val, select=None, is_nonexistent=False):
    select = select or where_col
    return {
        "table": table,
        "where_col": where_col,
        "where_val": where_val,
        "select": select,
        "is_nonexistent": is_nonexistent,
        "sql": f"SELECT {select} FROM {table} WHERE {where_col} = ?",
    }


# --- literal_in_question: 텍스트 -------------------------------------------


def test_literal_in_question_text_exact():
    assert qg.literal_in_question("who is marilyn", "marilyn", "text")


def test_literal_in_question_text_natural_plural_allowed():
    # 자연스러운 복수형(colander -> colanders)은 허용해야 함 (Qwen3 파일럿에서 실제로 나온 케이스)
    assert qg.literal_in_question(
        "how many colanders are left in stock", "colander", "text"
    )


def test_literal_in_question_text_embedded_in_different_word_rejected():
    # "ana"가 "anastasia" 안에 우연히 들어있는 걸 리터럴 포함으로 오인하면 안 됨
    assert not qg.literal_in_question("tell me about anastasia", "ana", "text")


def test_literal_in_question_text_missing():
    assert not qg.literal_in_question("who is this customer", "marilyn", "text")


# --- literal_in_question: 숫자(float/int) 경계 처리 -------------------------


def test_literal_in_question_float_exact():
    assert qg.literal_in_question("the item priced 97.56", 97.56, "float")


def test_literal_in_question_float_trailing_sentence_period_allowed():
    # 문장을 끝내는 마침표를 소수점이 더 이어지는 것으로 오판하면 안 됨
    assert qg.literal_in_question("the item priced 97.56.", 97.56, "float")


def test_literal_in_question_float_embedded_in_longer_number_rejected():
    # 목표값 97.56이 197.56 안에 부분 문자열로 파묻혀 오탐되면 안 됨 (실제로 있었던 버그)
    assert not qg.literal_in_question("the item priced 197.56", 97.56, "float")


def test_literal_in_question_float_extended_decimal_rejected():
    # 97.56이 97.567처럼 소수부가 더 이어지는 다른 값의 접두부일 때 오탐되면 안 됨
    assert not qg.literal_in_question("the item priced 97.567", 97.56, "float")


def test_literal_in_question_float_stripped_trailing_zero_allowed():
    # 100.00처럼 소수부가 0이면 "100"만 써도 통과해야 함
    assert qg.literal_in_question("the item priced 100", 100.00, "float")


def test_literal_in_question_int_exact():
    assert qg.literal_in_question("customer 1653", 1653, "int")


def test_literal_in_question_int_embedded_in_longer_number_rejected():
    assert not qg.literal_in_question("customer 11653", 1653, "int")


# --- check_question ----------------------------------------------------------


def test_check_question_passes_clean_question():
    entry = _entry("customers", "name", "marilyn")
    assert qg.check_question("what city is marilyn from", entry) == []


def test_check_question_fails_uppercase():
    entry = _entry("customers", "name", "marilyn")
    fails = qg.check_question("What city is marilyn from", entry)
    assert any("소문자" in f for f in fails)


def test_check_question_fails_missing_literal():
    entry = _entry("customers", "name", "marilyn")
    fails = qg.check_question("what city is this customer from", entry)
    assert any("리터럴" in f for f in fails)


def test_check_question_fails_spelled_out_number():
    entry = _entry("items", "stock", 20)
    fails = qg.check_question("about twenty units, stock is 20", entry)
    assert any("숫자" in f for f in fails)


def test_check_question_fails_snake_case_column_name():
    entry = _entry("items", "item_id", 3009)
    fails = qg.check_question("what is the item_id 3009", entry)
    assert any("컬럼명" in f for f in fails)


def test_check_question_fails_known_synonym_swap():
    entry = _entry("items", "item_name", "laptop")
    fails = qg.check_question("what is the price of this computer, the laptop", entry)
    assert any("동의어" in f for f in fails)


def test_check_question_allows_synonym_word_when_it_is_the_real_literal():
    # 금지어 자체가 실제 정답 리터럴인 경우(상품명이 진짜 "couch")는 실패로 잡으면 안 됨
    entry = _entry("items", "item_name", "couch")
    assert qg.check_question("tell me about the couch", entry) == []


# --- soft_flags ----------------------------------------------------------------


def test_soft_flags_stock_quantity_confusion():
    entry = _entry("items", "stock", 40, select="stock")
    flags = qg.soft_flags("how many were ordered for this item", entry)
    assert any("혼동" in f for f in flags)


def test_soft_flags_stock_no_confusion_when_worded_clearly():
    # select를 where_col과 다르게 둬서(규칙10과 섞이지 않게) stock/quantity 혼동만 단독으로 검사
    entry = _entry("items", "stock", 40, select="item_name")
    assert qg.soft_flags("how many are left in stock", entry) == []


def test_soft_flags_rule10_tautological_flagged():
    entry = _entry("items", "price", 97.56, select="price")
    flags = qg.soft_flags("what price is the item priced 97.56", entry)
    assert any("규칙 10" in f for f in flags)


def test_soft_flags_rule10_existence_framing_not_flagged():
    entry = _entry("items", "price", 97.56, select="price")
    flags = qg.soft_flags("is there an item priced at 97.56", entry)
    assert not any("규칙 10" in f for f in flags)


def test_soft_flags_rule10_not_applied_when_select_differs_from_where():
    # select와 where_col이 다르면 규칙10 자체가 해당 없음
    entry = _entry("items", "price", 97.56, select="item_name")
    assert qg.soft_flags("what price is the item priced 97.56", entry) == []


# --- find_intra_group_duplicates (규칙6: 같은 id 내 5문장은 서로 달라야 함) ---


def test_find_intra_group_duplicates_detects_exact_repeat():
    qs = ["a", "b", "a", "c", "d"]
    assert qg.find_intra_group_duplicates(qs) == {2}


def test_find_intra_group_duplicates_detects_punctuation_only_repeat():
    qs = ["what is the price?", "what is the price", "a different one"]
    assert qg.find_intra_group_duplicates(qs) == {1}


def test_find_intra_group_duplicates_no_false_positive_on_distinct_questions():
    qs = ["a", "b", "c", "d", "e"]
    assert qg.find_intra_group_duplicates(qs) == set()


# --- find_cross_sql_question_conflicts (전역 라벨 충돌 검사) -------------------


def test_find_cross_sql_question_conflicts_detects_conflict():
    pairs = [
        {
            "question": "what item costs 10",
            "sql": "SELECT * FROM items WHERE price = 10",
        },
        {
            "question": "what item costs 10",
            "sql": "SELECT item_name FROM items WHERE price = 10",
        },
    ]
    conflicts = qg.find_cross_sql_question_conflicts(pairs)
    assert len(conflicts) == 1
    assert conflicts[0]["question"] == "what item costs 10"
    assert len(conflicts[0]["conflicting_sqls"]) == 2


def test_find_cross_sql_question_conflicts_ignores_true_duplicates():
    pairs = [
        {
            "question": "what item costs 10",
            "sql": "SELECT * FROM items WHERE price = 10",
        },
        {
            "question": "what item costs 10",
            "sql": "SELECT * FROM items WHERE price = 10",
        },
    ]
    assert qg.find_cross_sql_question_conflicts(pairs) == []


# --- stratified_sample ------------------------------------------------------


def test_stratified_sample_preserves_group_ratio():
    entries = [{"table": "a", "where_col": "x"}] * 80 + [
        {"table": "b", "where_col": "y"}
    ] * 20
    picked = qg.stratified_sample(entries, 10, seed=0)
    groups = {}
    for _, e in picked:
        key = (e["table"], e["where_col"])
        groups[key] = groups.get(key, 0) + 1
    assert groups.get(("a", "x"), 0) == 8
    assert groups.get(("b", "y"), 0) == 2


def test_stratified_sample_deterministic_with_same_seed():
    entries = [{"table": "a", "where_col": "x", "i": i} for i in range(50)]
    p1 = qg.stratified_sample(entries, 10, seed=42)
    p2 = qg.stratified_sample(entries, 10, seed=42)
    assert p1 == p2


# --- small_model_rules_blurb (작은 모델용 프롬프트 문장 수 조절) ---------------


def test_small_model_rules_blurb_respects_question_count():
    blurb = qg.small_model_rules_blurb(3)
    assert "write 3 different" in blurb
    assert "1. casual/conversational style" in blurb
    assert "4." not in blurb.split("Rules")[0]  # 3개 요청이면 슬롯이 3개까지만 나열됨


def test_small_model_rules_blurb_rejects_too_many_slots():
    import pytest

    with pytest.raises(ValueError):
        qg.small_model_rules_blurb(len(qg.QUESTION_STYLE_SLOTS) + 1)


# --- _load_responses 견고성 (형식 오류/중복 id를 죽지 않고 보고) ----------------


def test_load_responses_handles_malformed_and_duplicate_ids(tmp_path):
    resp_dir = tmp_path / "responses"
    resp_dir.mkdir()
    (resp_dir / "batch_01.json").write_text(
        json.dumps(
            [
                {"id": 1, "questions": ["a", "b"]},
                {
                    "id": 1,
                    "questions": ["c", "d"],
                },  # 중복 id -- 나중 값으로 덮어써지되 경고를 남겨야 함
                {
                    "no_id_field": True
                },  # 형식 오류 -- 이 항목만 건너뛰고 나머지는 처리돼야 함
            ]
        ),
        encoding="utf-8",
    )
    (resp_dir / "batch_02.json").write_text(
        "not a json array, {broken", encoding="utf-8"
    )

    responses, missing = qg._load_responses(str(tmp_path))

    assert responses[1] == ["c", "d"]
    assert any("중복" in m for m in missing)
    assert any("형식 오류" in m for m in missing)
    assert any("파싱 실패" in m for m in missing)
