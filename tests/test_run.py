import sqlite3

import pytest
import torch

from src import run as rn
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
    assert rn.predict(model, "  What City Is Ashley In?  ", TOK, None) == sql
    assert prompts[0] == [BOS_ID] + TOK.encode("what city is ashley in?") + [SEP_ID]


@pytest.mark.unit
def test_predict_rejects_too_long_question():
    """프롬프트가 max_seq_len 이상이면 ValueError"""
    model, _ = scripted_model("x")
    with pytest.raises(ValueError):
        rn.predict(model, "a" * MAX_LEN, TOK, None)


@pytest.mark.unit
def test_predict_returns_none_on_broken_bytes():
    """깨진 UTF-8 을 생성하면 None"""
    model, _ = scripted_model([0xFF])
    assert rn.predict(model, "hi", TOK, None) is None


# ===========================================================================
# run_sql / answer
# ===========================================================================

@pytest.mark.unit
def test_run_sql_prints_rows(con, capsys):
    """결과 행 수·컬럼명·행 값을 출력"""
    rn.run_sql(con, "SELECT * FROM customers WHERE city = 'paris'")
    out = capsys.readouterr().out
    assert "결과 1행" in out and "customer_id, name, city" in out and "1001 | ashley | paris" in out


@pytest.mark.unit
def test_run_sql_truncates_long_results(con, capsys, monkeypatch):
    """MAX_ROWS 를 넘는 행은 생략 안내만 출력"""
    monkeypatch.setattr(rn, "MAX_ROWS", 1)
    rn.run_sql(con, "SELECT name FROM customers")
    out = capsys.readouterr().out
    assert "결과 2행" in out and "1행 더 있음" in out


@pytest.mark.unit
def test_run_sql_reports_error(con, capsys):
    """실행 오류는 예외 없이 메시지로 출력"""
    rn.run_sql(con, "SELECT nope FROM customers")
    assert "실행 오류" in capsys.readouterr().out


@pytest.mark.unit
def test_answer_flags_non_stage1_sql(capsys):
    """1단계 형식이 아닌 SQL 은 표시를 붙이고, con=None 이면 실행하지 않음"""
    model, _ = scripted_model("SELECT city FROM customers")
    rn.answer("hi", model, TOK, None, None)
    out = capsys.readouterr().out
    assert "SQL: SELECT city FROM customers" in out and "[1단계 SQL 형식 아님]" in out
    assert "결과" not in out


@pytest.mark.unit
def test_answer_executes_stage1_sql(con, capsys):
    """1단계 SQL 이면 표시 없이 출력하고 실행 결과까지 보여줌"""
    model, _ = scripted_model("SELECT city FROM customers WHERE name = 'henry'")
    rn.answer("where does henry live?", model, TOK, None, con)
    out = capsys.readouterr().out
    assert "형식 아님" not in out and "결과 1행" in out and "rome" in out


@pytest.mark.unit
def test_answer_handles_undecodable_output(capsys):
    """디코딩 불가 출력은 안내 메시지"""
    model, _ = scripted_model([0xFF])
    rn.answer("hi", model, TOK, None, None)
    assert "디코딩 불가" in capsys.readouterr().out


# ===========================================================================
# 입력·모델 해석 (이슈 #26 동작 명세)
# ===========================================================================

def _write_json(path, obj):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.mark.unit
def test_resolve_model_file_folder_and_default(tmp_path):
    """파일은 그대로, 폴더는 model.pt > best.pt, 생략하면 단계의 배포 모델"""
    paths = rn.config.stage_paths(1)
    assert rn.resolve_model(None, paths) == paths.model_path
    (tmp_path / "best.pt").write_bytes(b"")
    assert rn.resolve_model(str(tmp_path), paths) == tmp_path / "best.pt"
    (tmp_path / "model.pt").write_bytes(b"")
    assert rn.resolve_model(str(tmp_path), paths) == tmp_path / "model.pt"
    assert rn.resolve_model(str(tmp_path / "x.pt"), paths) == tmp_path / "x.pt"
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        rn.resolve_model(str(tmp_path / "empty"), paths)


@pytest.mark.unit
def test_resolve_input_kinds(tmp_path):
    """생략 / 질문 문자열 / 문자열 목록 JSON / 쌍 JSON(+같은 폴더 sql.json) / 평가셋 폴더 / 평가셋 이름"""
    pairs = [{"question": "q1", "sql": "S1"}]
    _write_json(tmp_path / "eval" / "mini" / "pairs.json", pairs)
    _write_json(tmp_path / "eval" / "mini" / "sql.json", [{"sql": "S1", "table": "t"}])
    _write_json(tmp_path / "qs.json", ["what city is ashley in?", "hi"])
    eval_dir = tmp_path / "eval"

    assert rn.resolve_input(None, eval_dir) == {"kind": "none"}
    assert rn.resolve_input("what city is ashley in?", eval_dir) == {"kind": "question", "question": "what city is ashley in?"}

    q = rn.resolve_input(str(tmp_path / "qs.json"), eval_dir)
    assert q["kind"] == "pairs" and q["name"] == "qs" and q["meta"] is None
    assert q["pairs"] == [{"question": "what city is ashley in?"}, {"question": "hi"}]

    for value in (str(eval_dir / "mini" / "pairs.json"), str(eval_dir / "mini"), "mini"):
        r = rn.resolve_input(value, eval_dir)
        assert r["kind"] == "pairs" and r["name"] == "mini" and r["pairs"] == pairs and r["meta"] == {"S1": {"sql": "S1", "table": "t"}}

    with pytest.raises(FileNotFoundError):
        rn.resolve_input(str(tmp_path / "missing.json"), eval_dir)


# ===========================================================================
# main: 인자 조합별 동작 (작은 무작위 모델로 동작 경로만 확인)
# ===========================================================================

@pytest.fixture
def setup(tmp_path):
    """작은 모델 폴더(model.pt + tokenizer.json), 평가셋 2개, DB"""
    import json
    cfg = dict(vocab_size=320, dim=32, num_layers=1, num_heads=2, ffn_dim=64, max_seq_len=64, padding_idx=PAD_ID)
    torch.manual_seed(0)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.save({"format_version": 1, "stage": 1, "model_cfg": cfg,
                "model": TextToSQLModel(**cfg).state_dict()}, model_dir / "model.pt")
    TOK.save(model_dir / "tokenizer.json")
    db = tmp_path / "shop.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE customers (customer_id INTEGER, name TEXT, city TEXT)")
    c.execute("INSERT INTO customers VALUES (1001, 'ashley', 'paris')")
    c.commit()
    c.close()
    sql = "SELECT city FROM customers WHERE name = 'ashley'"
    for name in ("a", "b"):
        _write_json(tmp_path / "eval" / name / "pairs.json", [{"question": "what city is ashley in?", "sql": sql}])
        _write_json(tmp_path / "eval" / name / "sql.json", [{"sql": sql, "table": "customers", "where_col": "name"}])
    _write_json(tmp_path / "qs.json", ["what city is ashley in?"])
    base = ["--model", str(model_dir), "--db", str(db), "--eval-dir", str(tmp_path / "eval"), "--cpu", "--show", "0"]
    return tmp_path, base


@pytest.mark.integration
def test_main_question_with_evaluation_falls_back_to_test(setup, capsys):
    """질문 문자열 + --evaluation: 정답 없음 안내 후 단건 추론"""
    tmp, base = setup
    assert rn.main(base + ["--input", "what city is ashley in?", "--evaluation"]) == 0
    out = capsys.readouterr().out
    assert "--test(단건 추론)로 진행" in out and "SQL:" in out


@pytest.mark.integration
def test_main_json_without_gold_saves_predictions(setup):
    """정답 없는 JSON + --evaluation: 배치 추론으로 바꿔 예측 파일 저장"""
    import json
    tmp, base = setup
    assert rn.main(base + ["--input", str(tmp / "qs.json"), "--evaluation", "--output", str(tmp / "out" / "p.json")]) == 0
    recs = json.loads((tmp / "out" / "p.json").read_text(encoding="utf-8"))
    assert recs[0]["question"] == "what city is ashley in?" and "pred" in recs[0] and "sql" not in recs[0]


@pytest.mark.integration
def test_main_eval_set_default_mode_is_evaluation(setup):
    """정답 있는 평가셋을 모드 없이 주면 채점, --output 폴더에 eval_<이름>.json"""
    import json
    tmp, base = setup
    assert rn.main(base + ["--input", "a", "--output", str(tmp / "out")]) == 0
    r = json.loads((tmp / "out" / "eval_a.json").read_text(encoding="utf-8"))
    assert r["set"] == "a" and r["overall"]["n"] == 1 and "by_where_col" in r


@pytest.mark.integration
def test_main_eval_set_with_test_flag_only_predicts(setup):
    """정답 있는 평가셋 + --test: 채점 없이 예측만 저장 (정답도 함께 남김)"""
    import json
    tmp, base = setup
    assert rn.main(base + ["--input", "a", "--test", "--output", str(tmp / "out")]) == 0
    recs = json.loads((tmp / "out" / "a.json").read_text(encoding="utf-8"))
    assert set(recs[0]) == {"question", "pred", "stage_format", "sql"}


@pytest.mark.integration
def test_main_evaluation_without_input_runs_all_sets(setup):
    """입력 없이 --evaluation: 평가셋 전부, --output 이 .json 이면 한 파일에 묶어 저장"""
    import json
    tmp, base = setup
    assert rn.main(base + ["--evaluation", "--output", str(tmp / "all.json")]) == 0
    assert set(json.loads((tmp / "all.json").read_text(encoding="utf-8"))) == {"a", "b"}


@pytest.mark.integration
def test_main_list_and_build_sql(setup, capsys):
    """--list-sets, --build-sql 은 모델 없이 동작"""
    tmp, _ = setup
    assert rn.main(["--list-sets", "--eval-dir", str(tmp / "eval")]) == 0
    assert "a " in capsys.readouterr().out
    (tmp / "eval" / "a" / "sql.json").unlink()
    _write_json(tmp / "src_sql.json", [{"sql": "SELECT city FROM customers WHERE name = 'ashley'", "table": "customers"}])
    assert rn.main(["--build-sql", "a", "--sql", str(tmp / "src_sql.json"), "--eval-dir", str(tmp / "eval")]) == 0
    assert (tmp / "eval" / "a" / "sql.json").is_file()


@pytest.mark.integration
def test_main_eval_sets_selects_sets_and_out_alias(setup):
    """--eval-sets 로 고른 평가셋만, --out 은 --output 과 같다. --test 와 함께면 추론만"""
    import json
    tmp, base = setup
    assert rn.main(base + ["--eval-sets", "b", "--out", str(tmp / "o1")]) == 0
    assert [p.name for p in (tmp / "o1").iterdir()] == ["eval_b.json"]
    assert rn.main(base + ["--eval-sets", "a", "b", "--test", "--out", str(tmp / "p.json")]) == 0
    assert set(json.loads((tmp / "p.json").read_text(encoding="utf-8"))) == {"a", "b"}


@pytest.mark.unit
def test_main_eval_sets_rejects_unknown_or_with_input(setup):
    """없는 평가셋 이름, --input 과 함께 쓰기는 거부"""
    tmp, base = setup
    with pytest.raises(SystemExit):
        rn.main(base + ["--eval-sets", "nope"])
    with pytest.raises(SystemExit):
        rn.main(base + ["--eval-sets", "a", "--input", "b"])
