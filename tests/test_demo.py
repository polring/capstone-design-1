import json

import pytest

from src import config
from src import demo

SQL = demo.DEFAULT_SQL
TEXT = "where does ashley live?"


@pytest.fixture
def small_train(tmp_path):
    """학습 쌍 앞 200개만 담은 파일 (전체 2만여 쌍을 처리하면 느리다)"""
    pairs = json.loads((config.TRAIN_DIR / config.PAIRS_FILE).read_text(encoding="utf-8"))[:200]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(pairs), encoding="utf-8")
    return str(path)


@pytest.mark.integration
@pytest.mark.parametrize("argv, marker", [
    (["db"], "미등장 값"),
    (["db", "--sql", SQL], "WHERE 조건에 맞는 행"),
    (["sql"], "출력 SQL"),
    (["sql", "--sql", SQL], "분할: train"),
    (["questions"], "출력 질문"),
    (["questions", "--sql", SQL], "출력 질문 [train]"),
    (["clean", "--text", TEXT, "--sql", SQL], "출력: 통과"),
    (["name-swap", "--text", TEXT, "--sql", SQL], "출력: "),
    (["tokenizer", "--text", TEXT, "--sql", SQL], "디코딩"),
    (["embedding", "--text", TEXT], "RoPE"),
    (["block", "--text", TEXT, "--layer", "2"], "블록 forward 결과와 같음: True"),
    (["model", "--text", TEXT, "--sql", SQL], "teacher forcing loss"),
    (["generate", "--text", TEXT, "--sql", SQL], "정답 SQL 과 같음(EM)"),
    (["score", "--text", TEXT, "--sql", SQL], "다중 정답 EM"),
])
def test_demo_commands(argv, marker, capsys):
    """하위 명령마다 --text/--sql 이 있을 때와 없을 때 모두 커밋된 데이터·배포 모델로 동작한다"""
    assert demo.main(argv + ["--n", "1"]) == 0
    assert marker in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.parametrize("command, marker", [
    ("clean", "제거 대상"),
    ("name-swap", "통계"),
    ("dataset", "전체 길이 통계"),
])
def test_demo_train_file_commands(command, marker, small_train, capsys):
    """학습 쌍 전체를 읽는 하위 명령은 --train-files 로 다른 파일을 받는다"""
    assert demo.main([command, "--n", "1", "--train-files", small_train]) == 0
    assert marker in capsys.readouterr().out


@pytest.mark.integration
def test_demo_all_follows_one_example(capsys):
    """all: SQL 하나를 DB → 생성 정보 → 질문 → … → 생성 SQL 까지 따라가고, 블록 출력이 출력층 입력으로 이어진다"""
    assert demo.main(["all", "--sql", SQL]) == 0
    out = capsys.readouterr().out
    assert "분할: train" in out and ">>> 이후 단계는 질문" in out
    assert "model(ids) 를 한 번에 부른 결과와 같음: True" in out  # 단계별로 넘긴 블록 출력 = 한 번에 계산한 결과
    assert "정답 SQL 과 같음(EM): True" in out
    assert "EM (문자열 완전 일치)           : True" in out   # 생성 SQL 이 채점 단계로 넘어감


@pytest.mark.unit
def test_demo_score_given_prediction(capsys):
    """--pred 를 주면 모델 없이 채점만: * 대신 식별 컬럼을 고른 답은 EM 은 틀리고 다중 정답 EM 은 맞음"""
    gold = "SELECT * FROM customers WHERE name = 'harriet'"
    assert demo.main(["score", "--text", "do we have a customer named harriet", "--sql", gold,
                      "--pred", "SELECT name FROM customers WHERE name = 'harriet'"]) == 0
    out = capsys.readouterr().out
    assert "EM (문자열 완전 일치)           : False" in out and "다중 정답 EM                    : True" in out
    assert "['select']" in out


@pytest.mark.unit
def test_demo_clean_requires_text_and_sql_together():
    """clean·name-swap 은 --text 와 --sql 을 함께 주거나 둘 다 생략한다"""
    with pytest.raises(SystemExit):
        demo.main(["clean", "--text", TEXT])


@pytest.mark.unit
def test_demo_quiet_hides_explanations(capsys):
    """기본 출력에는 ※ 설명 줄이 있고, --quiet 면 설명 없이 값만 출력한다"""
    demo.main(["tokenizer", "--text", TEXT])
    assert "※" in capsys.readouterr().out
    demo.main(["tokenizer", "--text", TEXT, "--quiet"])
    out = capsys.readouterr().out
    assert "※" not in out and "디코딩" in out


@pytest.mark.integration
def test_demo_output_encodes_in_cp949():
    """Windows 에서 출력을 파일·파이프로 넘기면 cp949 로 인코딩된다. demo 출력에 cp949 에 없는 문자가 없어야 함"""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo.main(["all", "--sql", SQL])
    buf.getvalue().encode("cp949")
