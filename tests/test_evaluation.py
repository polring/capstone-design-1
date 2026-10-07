import json
import sqlite3
from collections import Counter

import pytest
import torch

from src import config
from src import evaluation as ev
from src.data.dataset import EOS_ID, PAD_ID, tokenize_pairs
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

TOK = bpe.Tokenizer(bpe.load_seed_tokens())  # 병합 없이 seed + 바이트 단위
VOCAB = 300


def tiny_model(max_seq_len: int = 48) -> TextToSQLModel:
    torch.manual_seed(0)
    return TextToSQLModel(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                          max_seq_len=max_seq_len, padding_idx=PAD_ID)


# ===========================================================================
# 채점 보조
# ===========================================================================

@pytest.mark.unit
def test_execute():
    """실행 결과를 순서 무관 multiset 으로 반환하고, 오류·None 은 None"""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (a INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (2,)])
    assert ev.execute(con, "SELECT a FROM t") == Counter({(1,): 1, (2,): 2})
    assert ev.execute(con, "SELECT a FROM t ORDER BY a DESC") == ev.execute(con, "SELECT a FROM t")
    assert ev.execute(con, "SELECT nope FROM t") is None
    assert ev.execute(con, None) is None


@pytest.mark.unit
def test_wilson_ci():
    """Wilson 95% 신뢰구간: 알려진 값, 경계, n=0 처리"""
    lo, hi = ev.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = ev.wilson_ci(100, 100)
    assert hi == pytest.approx(1.0) and 0.95 < lo < 1.0
    assert ev.wilson_ci(0, 0) == (0.0, 0.0)


@pytest.mark.unit
def test_summarize():
    """EM / 다중 정답 EM / 실행 정확도 비율 집계 검증"""
    records = [
        {"em": True, "em_multi": True, "exec_match": True},
        {"em": False, "em_multi": True, "exec_match": True},
        {"em": False, "em_multi": False, "exec_match": False},
        {"em": False, "em_multi": False, "exec_match": True},
    ]
    s = ev.summarize(records)
    assert s["n"] == 4 and s["em"] == 0.25 and s["em_multi"] == 0.5 and s["exec_acc"] == 0.75
    assert s["em_ci95"][0] <= 0.25 <= s["em_ci95"][1]


@pytest.mark.unit
def test_safe_decode():
    """정상 바이트는 복원, 깨진 UTF-8 이나 vocab 밖 ID 는 None"""
    assert ev.safe_decode(TOK.encode("SELECT x"), TOK) == "SELECT x"
    assert ev.safe_decode([0xFF], TOK) is None
    assert ev.safe_decode([VOCAB + 500], TOK) is None


@pytest.mark.unit
def test_split_example_roundtrip():
    """토큰화된 예시에서 (질문, 정답 SQL) 을 그대로 복원하는지 검증"""
    pair = {"question": "what city is ashley in?", "sql": "SELECT city FROM customers WHERE name = 'ashley'"}
    ex = tokenize_pairs([pair], TOK)[0]
    assert ev.split_example(ex, TOK) == (pair["question"], pair["sql"])


# ===========================================================================
# generate_batch
# ===========================================================================

def scripted_forward(script_by_first_token: dict[int, list[int]], prompt_lens: dict[int, int]):
    """forward 대체: 행의 두 번째 토큰(질문 첫 토큰)으로 행을 구분해 정해진 스크립트를 생성한다."""
    def forward(idx):
        logits = torch.zeros(idx.size(0), idx.size(1), VOCAB)
        for b in range(idx.size(0)):
            key = idx[b, 1].item()
            k = idx.size(1) - prompt_lens[key]
            script = script_by_first_token[key]
            logits[b, -1, script[k] if k < len(script) else EOS_ID] = 1.0
        return logits
    return forward


@pytest.mark.unit
def test_generate_batch_groups_by_prompt_length_and_keeps_order():
    """프롬프트 길이가 다른 예시를 섞어도 입력 순서대로, <eos> 전까지만 반환하는지 검증"""
    pairs = [{"question": "a", "sql": "x"}, {"question": "bbb", "sql": "x"}, {"question": "cc", "sql": "x"},
             {"question": "dd", "sql": "x"}]
    examples = tokenize_pairs(pairs, TOK)
    first = {ex["ids"][1]: ex["sql_start"] for ex in examples}
    scripts = {ord("a"): [65], ord("b"): [66, 66, 66], ord("c"): [67, 67], ord("d"): []}
    model = tiny_model()
    model.forward = scripted_forward(scripts, first)
    model.train()
    out = ev.generate_batch(model, examples, torch.device("cpu"), None, batch_size=1)
    assert out == [[65], [66, 66, 66], [67, 67], []]
    assert model.training, "generate_batch 후 원래 train/eval 상태로 돌아가야 함"


@pytest.mark.integration
def test_generate_batch_matches_single_generate():
    """실제 모델에서 배치 생성 결과가 예시별 model.generate 결과와 같아야 함 (오른쪽 패딩 + 길이별 묶음)"""
    pairs = [{"question": q, "sql": "x"} for q in ("hi", "hello there", "yo", "abc", "hello world")]
    examples = tokenize_pairs(pairs, TOK)
    model = tiny_model().eval()
    batch = ev.generate_batch(model, examples, torch.device("cpu"), None, max_new_tokens=8)
    for ex, got in zip(examples, batch):
        assert got == model.generate(ex["ids"][:ex["sql_start"]], eos_id=EOS_ID, max_new_tokens=8)


# ===========================================================================
# evaluate_set
# ===========================================================================

@pytest.fixture
def eval_data(tmp_path):
    """임시 평가 폴더: indist/pairs.json + indist/sql.json, 그리고 shop.db"""
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE customers (customer_id INTEGER, name TEXT, city TEXT, membership TEXT)")
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)",
                    [(1001, "ashley", "paris", "gold"), (1002, "henry", "rome", "basic")])
    con.commit()
    con.close()

    pairs = [
        {"question": "what city is ashley in?", "sql": "SELECT city FROM customers WHERE name = 'ashley'"},
        {"question": "which customer lives in rome?", "sql": "SELECT * FROM customers WHERE city = 'rome'"},
    ]
    (tmp_path / "indist").mkdir()
    (tmp_path / "indist" / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    meta = [{"sql": p["sql"], "table": "customers", "where_col": c} for p, c in zip(pairs, ("name", "city"))]
    (tmp_path / "indist" / "sql.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path, pairs, db


@pytest.mark.integration
def test_evaluate_set_scores_scripted_predictions(eval_data):
    """정답 1개 + 다중 정답 1개를 내는 모델로 EM / multi / exec 집계와 오답 분류를 검증"""
    eval_dir, pairs, db = eval_data
    preds = {pairs[0]["question"]: pairs[0]["sql"],
             pairs[1]["question"]: "SELECT name FROM customers WHERE city = 'rome'"}
    examples = tokenize_pairs(pairs, TOK)
    model = tiny_model(max_seq_len=128)

    def forward(idx):
        logits = torch.zeros(idx.size(0), idx.size(1), VOCAB)
        for b in range(idx.size(0)):
            ex = next(e for e in examples if e["ids"][:e["sql_start"]] == idx[b, :e["sql_start"]].tolist())
            question, _ = ev.split_example(ex, TOK)
            target = TOK.encode(preds[question]) + [EOS_ID]
            logits[b, -1, target[idx.size(1) - ex["sql_start"]]] = 1.0
        return logits
    model.forward = forward

    result = ev.evaluate_set(model, "indist", TOK, torch.device("cpu"), None,
                             eval_dir=eval_dir, db_path=db)
    o = result["overall"]
    assert o["n"] == 2 and o["em"] == 0.5 and o["em_multi"] == 1.0
    assert o["exec_acc"] == 0.5  # SELECT name 과 SELECT * 은 결과가 다르다
    assert result["error_parts"] == {"select": 1}
    assert result["n_select_ambiguous"] == 1
    assert set(result["by_where_col"]) == {"customers.name", "customers.city"}
    assert [f["question"] for f in result["failures"]] == [pairs[1]["question"]]


@pytest.mark.unit
def test_evaluate_set_missing_pairs_raises(tmp_path):
    """평가 쌍 파일이 없으면 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        ev.evaluate_set(tiny_model(), "indist", TOK, torch.device("cpu"), None, eval_dir=tmp_path)


@pytest.mark.unit
def test_tokenizer_path_prefers_run_copy(tmp_path):
    """체크포인트 폴더의 tokenizer.json > 예전 bpe_merges.json > ckpt 에 적힌 경로 순으로 쓴다"""
    ckpt_path = tmp_path / "best.pt"
    assert ev.tokenizer_path_for(ckpt_path, {"tokenizer_path": "x.json"}) == ev.Path("x.json")
    assert ev.tokenizer_path_for(ckpt_path, {"merges_path": "old.json"}) == ev.Path("old.json")
    (tmp_path / "bpe_merges.json").write_text("[]", encoding="utf-8")
    assert ev.tokenizer_path_for(ckpt_path, {}) == tmp_path / "bpe_merges.json"
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert ev.tokenizer_path_for(ckpt_path, {}) == tmp_path / "tokenizer.json"


@pytest.mark.integration
def test_load_model_roundtrip(tmp_path):
    """train.py 형식 체크포인트(model + model_cfg)를 같은 출력의 모델로 다시 불러오는지 검증"""
    cfg = dict(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64, max_seq_len=48, padding_idx=PAD_ID)
    torch.manual_seed(0)
    model = TextToSQLModel(**cfg).eval()
    torch.save({"model": model.state_dict(), "model_cfg": cfg}, tmp_path / "best.pt")
    loaded, ckpt = ev.load_model(tmp_path / "best.pt", torch.device("cpu"))
    idx = torch.randint(0, 256, (1, 8))
    assert not loaded.training
    assert torch.allclose(model(idx), loaded(idx))


# ===========================================================================
# 평가셋 폴더 (pairs.json + sql.json)
# ===========================================================================

def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


PAIRS = [{"question": "q1", "sql": "S1"}, {"question": "q2", "sql": "S1"}, {"question": "q3", "sql": "S2"}]
SQL_ROWS = [{"sql": "S0", "table": "t"}, {"sql": "S2", "table": "t"}, {"sql": "S1", "table": "t"}]


@pytest.mark.unit
def test_list_sets_only_folders_with_pairs(tmp_path):
    """pairs.json 이 있는 하위 폴더만 평가셋으로 잡는다"""
    _write(tmp_path / "b" / "pairs.json", PAIRS)
    _write(tmp_path / "a" / "pairs.json", PAIRS)
    (tmp_path / "empty").mkdir()
    assert ev.list_sets(tmp_path) == ["a", "b"]
    assert ev.list_sets(tmp_path / "missing") == []


@pytest.mark.unit
def test_write_sql_meta_keeps_used_sql_in_source_order(tmp_path):
    """sql.json 은 쌍에 쓰인 SQL 만, 원래 SQL 파일 순서대로 담는다"""
    _write(tmp_path / "x" / "pairs.json", PAIRS)
    _write(tmp_path / "sql.json", SQL_ROWS)
    ev.write_sql_meta("x", tmp_path / "sql.json", tmp_path)
    pairs, meta = ev.load_eval_set("x", tmp_path)
    assert pairs == PAIRS
    assert list(meta) == ["S2", "S1"]


@pytest.mark.unit
def test_build_sql_meta_missing_sql_raises(tmp_path):
    """쌍의 SQL 이 SQL 파일에 없으면 오류"""
    _write(tmp_path / "pairs.json", PAIRS)
    _write(tmp_path / "sql.json", SQL_ROWS[:2])
    with pytest.raises(ValueError):
        ev.build_sql_meta(tmp_path / "pairs.json", tmp_path / "sql.json")


@pytest.mark.unit
def test_load_eval_set_stale_sql_meta_raises(tmp_path):
    """sql.json 을 만든 뒤 pairs.json 에 새 SQL 이 들어오면 불러올 때 오류"""
    _write(tmp_path / "x" / "pairs.json", PAIRS)
    _write(tmp_path / "x" / "sql.json", SQL_ROWS[2:])
    with pytest.raises(ValueError):
        ev.load_eval_set("x", tmp_path)


@pytest.mark.unit
def test_committed_eval_sets_are_consistent():
    """저장소의 평가셋 폴더마다 sql.json 이 pairs.json 의 SQL 을 모두 담고 있어야 함"""
    names = ev.list_sets()
    assert {"indist", "holdout"} <= set(names)
    for name in names:
        pairs, meta = ev.load_eval_set(name)
        assert len(meta) == len({p["sql"] for p in pairs})


@pytest.mark.unit
def test_eval_sets_outside_train_dir():
    """평가셋 폴더는 학습 폴더(= 학습 데이터·BPE 코퍼스) 밖에 있어야 함"""
    train = config.TRAIN_DIR.resolve()
    assert train not in config.EVAL_DIR.resolve().parents and train != config.EVAL_DIR.resolve()


@pytest.mark.unit
def test_load_rules_provides_stage_interface():
    """현재 단계의 SQL 규칙 모듈이 evaluation·predict 가 쓰는 이름을 모두 제공해야 함"""
    rules = ev.load_rules(config.STAGE)
    assert rules.STAGE == config.STAGE
    for name in ("parse_sql", "error_parts", "select_ambiguous", "multi_match", "MULTI_NOTE"):
        assert hasattr(rules, name), name


@pytest.mark.unit
def test_default_out_dir_stays_under_runs():
    """평가 결과는 커밋 폴더(models/)에 쓰지 않는다: runs/ 안 체크포인트는 그 폴더, 그 밖은 RELEASE_EVAL_DIR"""
    run_ckpt = config.RUNS_DIR / "some_run" / "best.pt"
    assert ev.default_out_dir(run_ckpt) == run_ckpt.parent
    assert ev.default_out_dir(config.MODEL_PATH) == config.RELEASE_EVAL_DIR
    assert ev.default_out_dir("runs/stage1_old_run/best.pt") == ev.Path("runs/stage1_old_run")
