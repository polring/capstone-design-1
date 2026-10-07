import sqlite3

import pytest
import torch

from src import predict as pr
from src.data.dataset import BOS_ID, EOS_ID, PAD_ID, SEP_ID
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

VOCAB, MAX_LEN = 300, 64
TOK = bpe.Tokenizer(bpe.load_seed_tokens())  # 병합 없이 seed + 바이트 단위


def scripted_model(output: str | list[int]) -> tuple[TextToSQLModel, list]:
    """프롬프트 뒤에 output 을 생성하고 <eos> 를 내는 모델. 받은 프롬프트를 기록한다."""
    torch.manual_seed(0)
    model = TextToSQLModel(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                           max_seq_len=MAX_LEN, padding_idx=PAD_ID).eval()
    script = (TOK.encode(output) if isinstance(output, str) else output) + [EOS_ID]
    prompts = []

    def forward(idx):
        row = idx[0].tolist()
        plen = row.index(SEP_ID) + 1
        if not prompts:
            prompts.append(row[:plen])
        logits = torch.zeros(1, idx.size(1), VOCAB)
        logits[0, -1, script[len(row) - plen]] = 1.0
        return logits
    model.forward = forward
    return model, prompts


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE customers (customer_id INTEGER, name TEXT, city TEXT)")
    c.executemany("INSERT INTO customers VALUES (?, ?, ?)", [(1001, "ashley", "paris"), (1002, "henry", "rome")])
    yield c
    c.close()


# ===========================================================================
# predict
# ===========================================================================

@pytest.mark.unit
def test_predict_builds_lowercased_prompt_and_decodes():
    """질문을 소문자·strip 해서 <bos> question <sep> 프롬프트로 넣고, 생성 결과를 SQL 문자열로 복원"""
    sql = "SELECT city FROM customers WHERE name = 'ashley'"
    model, prompts = scripted_model(sql)
    assert pr.predict(model, "  What City Is Ashley In?  ", TOK, None) == sql
    assert prompts[0] == [BOS_ID] + TOK.encode("what city is ashley in?") + [SEP_ID]


@pytest.mark.unit
def test_predict_rejects_too_long_question():
    """프롬프트가 max_seq_len 이상이면 ValueError"""
    model, _ = scripted_model("x")
    with pytest.raises(ValueError):
        pr.predict(model, "a" * MAX_LEN, TOK, None)


@pytest.mark.unit
def test_predict_returns_none_on_broken_bytes():
    """깨진 UTF-8 을 생성하면 None"""
    model, _ = scripted_model([0xFF])
    assert pr.predict(model, "hi", TOK, None) is None


# ===========================================================================
# run_sql / answer
# ===========================================================================

@pytest.mark.unit
def test_run_sql_prints_rows(con, capsys):
    """결과 행 수·컬럼명·행 값을 출력"""
    pr.run_sql(con, "SELECT * FROM customers WHERE city = 'paris'")
    out = capsys.readouterr().out
    assert "결과 1행" in out and "customer_id, name, city" in out and "1001 | ashley | paris" in out


@pytest.mark.unit
def test_run_sql_truncates_long_results(con, capsys, monkeypatch):
    """MAX_ROWS 를 넘는 행은 생략 안내만 출력"""
    monkeypatch.setattr(pr, "MAX_ROWS", 1)
    pr.run_sql(con, "SELECT name FROM customers")
    out = capsys.readouterr().out
    assert "결과 2행" in out and "1행 더 있음" in out


@pytest.mark.unit
def test_run_sql_reports_error(con, capsys):
    """실행 오류는 예외 없이 메시지로 출력"""
    pr.run_sql(con, "SELECT nope FROM customers")
    assert "실행 오류" in capsys.readouterr().out


@pytest.mark.unit
def test_answer_flags_non_stage1_sql(capsys):
    """1단계 형식이 아닌 SQL 은 표시를 붙이고, con=None 이면 실행하지 않음"""
    model, _ = scripted_model("SELECT city FROM customers")
    pr.answer("hi", model, TOK, None, None)
    out = capsys.readouterr().out
    assert "SQL: SELECT city FROM customers" in out and "[1단계 SQL 형식 아님]" in out
    assert "결과" not in out


@pytest.mark.unit
def test_answer_executes_stage1_sql(con, capsys):
    """1단계 SQL 이면 표시 없이 출력하고 실행 결과까지 보여줌"""
    model, _ = scripted_model("SELECT city FROM customers WHERE name = 'henry'")
    pr.answer("where does henry live?", model, TOK, None, con)
    out = capsys.readouterr().out
    assert "형식 아님" not in out and "결과 1행" in out and "rome" in out


@pytest.mark.unit
def test_answer_handles_undecodable_output(capsys):
    """디코딩 불가 출력은 안내 메시지"""
    model, _ = scripted_model([0xFF])
    pr.answer("hi", model, TOK, None, None)
    assert "디코딩 불가" in capsys.readouterr().out
