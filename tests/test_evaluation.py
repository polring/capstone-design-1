import json
import sqlite3
from collections import Counter

import pytest
import torch

from src import evaluation as ev
from src.data.dataset import EOS_ID, PAD_ID, tokenize_pairs
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

ID_TO_BYTES = bpe.id_to_bytes_from_merges([])
VOCAB = 300


def tiny_model(max_seq_len: int = 48) -> TextToSQLModel:
    torch.manual_seed(0)
    return TextToSQLModel(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                          max_seq_len=max_seq_len, padding_idx=PAD_ID)


# ===========================================================================
# parse_sql / error_parts
# ===========================================================================

@pytest.mark.unit
def test_parse_sql_stage1():
    """1단계 SQL 을 select / table / where_col / literal 로 분해하는지 검증"""
    assert ev.parse_sql("SELECT city FROM customers WHERE name = 'ashley'") == {
        "select": "city", "table": "customers", "where_col": "name", "literal": "'ashley'"}
    assert ev.parse_sql("SELECT * FROM items WHERE price = 14.46")["select"] == "*"


@pytest.mark.unit
@pytest.mark.parametrize("sql", [None, "", "SELECT city FROM customers", "DROP TABLE items"])
def test_parse_sql_rejects_non_stage1(sql):
    """1단계 문법이 아니거나 None 이면 None"""
    assert ev.parse_sql(sql) is None


@pytest.mark.unit
@pytest.mark.parametrize("pred, expected", [
    ("SELECT city FROM customers WHERE name = 'ashley'", []),
    ("SELECT city FROM customers WHERE name = 'ashly'", ["literal"]),
    ("SELECT * FROM customers WHERE name = 'ashley'", ["select"]),
    ("SELECT city FROM orders WHERE customer_id = 'ashley'", ["table", "where_col"]),
    (None, ["malformed"]),
    ("SELECT city FROM", ["malformed"]),
])
def test_error_parts(pred, expected):
    """정답과 다른 부분(table / where_col / select / literal)을 정확히 분류하는지 검증"""
    assert ev.error_parts("SELECT city FROM customers WHERE name = 'ashley'", pred) == expected


# ===========================================================================
# select_ambiguous / multi_match (다중 정답 EM)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.parametrize("question, gold, expected", [
    # 개체를 묻지만 이름인지 행 전체인지 단서가 없음
    ("which customer lives in paris?", "SELECT * FROM customers WHERE city = 'paris'", True),
    # "all" → 행 전체 단서
    ("show all details of customers in paris", "SELECT * FROM customers WHERE city = 'paris'", False),
    # "name" → 식별 컬럼 단서
    ("what's the name of the customer in paris?", "SELECT name FROM customers WHERE city = 'paris'", False),
    # SELECT 가 * / 식별 컬럼이 아니면 대상 아님
    ("what city is ashley in?", "SELECT city FROM customers WHERE name = 'ashley'", False),
    # 존재 확인형 (SELECT == WHERE 컬럼) 은 제외
    ("is there a customer named ashley?", "SELECT name FROM customers WHERE name = 'ashley'", False),
    # WHERE 조건 표현("named ashley")은 SELECT 단서로 보지 않음
    ("pull up the customer named ashley", "SELECT * FROM customers WHERE name = 'ashley'", True),
    # orders: "item id 3155" 는 WHERE 조건 표현이라 ID 단서가 아님
    ("orders for item id 3155", "SELECT * FROM orders WHERE item_id = 3155", True),
    ("order numbers for item 3155", "SELECT order_id FROM orders WHERE item_id = 3155", False),
    ("anything", "not sql", False),
])
def test_select_ambiguous(question, gold, expected):
    """SELECT * ↔ 식별 컬럼 모호 문항 판정 규칙 검증"""
    assert ev.select_ambiguous(question, gold) is expected


@pytest.mark.unit
def test_multi_match():
    """모호 문항에서는 SELECT 만 * ↔ 식별 컬럼으로 다른 예측도 정답, 그 외는 오답"""
    q, gold = "which customer lives in paris?", "SELECT * FROM customers WHERE city = 'paris'"
    assert ev.multi_match(q, gold, gold)
    assert ev.multi_match(q, gold, "SELECT name FROM customers WHERE city = 'paris'")
    assert not ev.multi_match(q, gold, "SELECT city FROM customers WHERE city = 'paris'")
    assert not ev.multi_match(q, gold, "SELECT name FROM customers WHERE city = 'rome'")
    assert not ev.multi_match(q, gold, None)
    # 단서가 있는 문항은 완전 일치만 정답
    q2 = "show all details of customers in paris"
    assert not ev.multi_match(q2, gold, "SELECT name FROM customers WHERE city = 'paris'")


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
    assert ev.safe_decode(bpe.encode("SELECT x", []), ID_TO_BYTES) == "SELECT x"
    assert ev.safe_decode([0xFF], ID_TO_BYTES) is None
    assert ev.safe_decode([VOCAB + 500], ID_TO_BYTES) is None


@pytest.mark.unit
def test_split_example_roundtrip():
    """토큰화된 예시에서 (질문, 정답 SQL) 을 그대로 복원하는지 검증"""
    pair = {"question": "what city is ashley in?", "sql": "SELECT city FROM customers WHERE name = 'ashley'"}
    ex = tokenize_pairs([pair], [])[0]
    assert ev.split_example(ex, ID_TO_BYTES) == (pair["question"], pair["sql"])


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
    examples = tokenize_pairs(pairs, [])
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
    examples = tokenize_pairs(pairs, [])
    model = tiny_model().eval()
    batch = ev.generate_batch(model, examples, torch.device("cpu"), None, max_new_tokens=8)
    for ex, got in zip(examples, batch):
        assert got == model.generate(ex["ids"][:ex["sql_start"]], eos_id=EOS_ID, max_new_tokens=8)


# ===========================================================================
# evaluate_set
# ===========================================================================

@pytest.fixture
def eval_data(tmp_path, monkeypatch):
    """임시 data 폴더: eval_indist_x/eval_pairs.json + sql_eval_indist.json + shop.db"""
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE customers (customer_id INTEGER, name TEXT, city TEXT, membership TEXT)")
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)",
                    [(1001, "ashley", "paris", "gold"), (1002, "henry", "rome", "basic")])
    con.commit()
    con.close()
    monkeypatch.setattr(ev, "DB_PATH", db)

    pairs = [
        {"question": "what city is ashley in?", "sql": "SELECT city FROM customers WHERE name = 'ashley'"},
        {"question": "which customer lives in rome?", "sql": "SELECT * FROM customers WHERE city = 'rome'"},
    ]
    (tmp_path / "eval_indist_x").mkdir()
    (tmp_path / "eval_indist_x" / "eval_pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    meta = [{"sql": p["sql"], "table": "customers", "where_col": c} for p, c in zip(pairs, ("name", "city"))]
    (tmp_path / "sql_eval_indist.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path, pairs


@pytest.mark.integration
def test_evaluate_set_scores_scripted_predictions(eval_data):
    """정답 1개 + 다중 정답 1개를 내는 모델로 EM / multi / exec 집계와 오답 분류를 검증"""
    data_dir, pairs = eval_data
    preds = {pairs[0]["question"]: pairs[0]["sql"],
             pairs[1]["question"]: "SELECT name FROM customers WHERE city = 'rome'"}
    examples = tokenize_pairs(pairs, [])
    model = tiny_model(max_seq_len=128)

    def forward(idx):
        logits = torch.zeros(idx.size(0), idx.size(1), VOCAB)
        for b in range(idx.size(0)):
            ex = next(e for e in examples if e["ids"][:e["sql_start"]] == idx[b, :e["sql_start"]].tolist())
            question, _ = ev.split_example(ex, ID_TO_BYTES)
            target = bpe.encode(preds[question], []) + [EOS_ID]
            logits[b, -1, target[idx.size(1) - ex["sql_start"]]] = 1.0
        return logits
    model.forward = forward

    result = ev.evaluate_set(model, "indist", [], ID_TO_BYTES, torch.device("cpu"), None, data_dir=data_dir)
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
        ev.evaluate_set(tiny_model(), "indist", [], ID_TO_BYTES, torch.device("cpu"), None, data_dir=tmp_path)


@pytest.mark.unit
def test_eval_set_globs_never_match_training_folders():
    """평가셋 glob 이 pilot* 학습 폴더를 가리키지 않고, holdout 과 holdout_qwen 이 서로 섞이지 않아야 함"""
    globs = {name: g for name, (g, _) in ev.EVAL_SETS.items()}
    assert all(not g.startswith("pilot") and g.endswith("/eval_pairs.json") for g in globs.values())
    import fnmatch
    assert not fnmatch.fnmatch("eval_qwen_holdout_merged/eval_pairs.json", globs["holdout"])
    assert not fnmatch.fnmatch("eval_holdout_merged/eval_pairs.json", globs["holdout_qwen"])


@pytest.mark.unit
def test_merges_path_prefers_run_copy(tmp_path):
    """체크포인트 폴더에 bpe_merges.json 사본이 있으면 그것을, 없으면 ckpt 의 merges_path 를 쓴다"""
    ckpt_path = tmp_path / "best.pt"
    assert ev.merges_path_for(ckpt_path, {"merges_path": "x.json"}) == ev.Path("x.json")
    (tmp_path / "bpe_merges.json").write_text("[]", encoding="utf-8")
    assert ev.merges_path_for(ckpt_path, {"merges_path": "x.json"}) == tmp_path / "bpe_merges.json"


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
