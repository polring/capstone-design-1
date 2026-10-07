import sys
import os

import pytest

# 프로젝트 루트 디렉터리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.generator import question_gen_auto as qga

pytestmark = pytest.mark.unit


def test_extract_json_array_valid_input():
    raw = '[{"id": 1, "questions": ["a", "b"]}]'
    assert qga.extract_json_array(raw) == [{"id": 1, "questions": ["a", "b"]}]


def test_extract_json_array_strips_think_tags():
    # Qwen3 계열의 기본 사고 모드 출력을 무시해야 함
    raw = (
        "<think>let me think about this...</think>" + '[{"id": 1, "questions": ["a"]}]'
    )
    assert qga.extract_json_array(raw) == [{"id": 1, "questions": ["a"]}]


def test_extract_json_array_strips_code_fences():
    raw = '```json\n[{"id": 1, "questions": ["a"]}]\n```'
    assert qga.extract_json_array(raw) == [{"id": 1, "questions": ["a"]}]


def test_extract_json_array_tolerates_trailing_garbage():
    # 실제 관찰된 패턴: 배열이 끝난 뒤 여분의 '}'가 하나 더 붙음
    raw = '[{"id": 1, "questions": ["a", "b"]}]}'
    assert qga.extract_json_array(raw) == [{"id": 1, "questions": ["a", "b"]}]


def test_extract_json_array_repairs_missing_array_close_on_every_item():
    # 실제 관찰된 패턴: 모든 항목의 "questions" 배열에 닫는 ']'가 통째로 빠짐
    broken = '[{"id":1,"questions":["a","b"},{"id":2,"questions":["c","d"}]'
    result = qga.extract_json_array(broken)
    assert result == [
        {"id": 1, "questions": ["a", "b"]},
        {"id": 2, "questions": ["c", "d"]},
    ]


def test_extract_json_array_repairs_missing_comma_between_objects():
    # 실제 관찰된 패턴: 객체 사이 콤마가 빠지고 바로 '{'가 이어짐
    broken = '[{"id":1,"questions":["a"]}{"id":2,"questions":["b"]}]'
    result = qga.extract_json_array(broken)
    assert result == [{"id": 1, "questions": ["a"]}, {"id": 2, "questions": ["b"]}]


def test_extract_json_array_regex_fallback_survives_swapped_brackets_and_missing_brace():
    # 실제 관찰된 가장 심한 패턴: 배열/객체 닫는 순서가 뒤바뀌고(']}' 대신 '}]')
    # 다음 객체를 여는 '{'까지 통째로 빠짐. 앞선 두 복구 단계로는 못 고치고,
    # id/questions 필드를 직접 정규식으로 뽑는 마지막 단계에서만 살아남아야 한다.
    broken = '[{"id":1,"questions":["a","b"}],"id":2,"questions":["c","d"}]'
    result = qga.extract_json_array(broken)
    assert result == [
        {"id": 1, "questions": ["a", "b"]},
        {"id": 2, "questions": ["c", "d"]},
    ]


def test_extract_json_array_gives_up_when_no_array_present():
    raw = "sorry, i can't help with that request."
    assert qga.extract_json_array(raw) is None


def test_extract_json_array_gives_up_on_request_error_marker():
    raw = "__REQUEST_ERROR__ timed out"
    assert qga.extract_json_array(raw) is None


def test_extract_json_array_does_not_silently_drop_items():
    # 전역 괄호 보정을 걸면 우연히 "항목 1개짜리" 유효 JSON으로 조기 파싱될 수 있는 입력.
    # _expected_item_count("id" 등장 횟수)와 다르면 그 결과를 버리고 다음 복구 단계로
    # 넘어가야 하며, 최종적으로는 두 항목 다 정확히 복구돼야 한다 (일부 항목이 조용히
    # 사라지는 거짓 성공을 허용하면 안 됨).
    broken = '[{"id":1,"questions":["a","b"}],"id":2,"questions":["c","d"}]'
    result = qga.extract_json_array(broken)
    assert result is not None
    assert len(result) == 2
