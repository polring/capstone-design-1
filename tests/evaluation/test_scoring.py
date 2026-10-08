import json
import sqlite3
from collections import Counter

import pytest
import torch

from src import config
from src.inference import generate
from src.evaluation import scoring
from src.data.dataset import EOS_ID, PAD_ID, tokenize_pairs
from src.models.model import TextToSQLModel
from src.tokenizer import bpe


TOK = bpe.Tokenizer(bpe.load_seed_tokens())  # 병합 없이 seed + 바이트 단위


VOCAB = 300


def tiny_model(max_seq_len: int = 48) -> TextToSQLModel:
    torch.manual_seed(0)
    return TextToSQLModel(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                          max_seq_len=max_seq_len, padding_idx=PAD_ID)


@pytest.mark.unit
def test_execute():
    """실행 결과를 순서 무관 multiset 으로 반환하고, 오류·None 은 None"""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (a INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (2,)])
    assert scoring.execute(con, "SELECT a FROM t") == Counter({(1,): 1, (2,): 2})
    assert scoring.execute(con, "SELECT a FROM t ORDER BY a DESC") == scoring.execute(con, "SELECT a FROM t")
    assert scoring.execute(con, "SELECT nope FROM t") is None
    assert scoring.execute(con, None) is None


@pytest.mark.unit
def test_wilson_ci():
    """Wilson 95% 신뢰구간: 알려진 값, 경계, n=0 처리"""
    lo, hi = scoring.wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = scoring.wilson_ci(100, 100)
    assert hi == pytest.approx(1.0) and 0.95 < lo < 1.0
    assert scoring.wilson_ci(0, 0) == (0.0, 0.0)


@pytest.mark.unit
def test_summarize():
    """EM / 다중 정답 EM / 실행 정확도 비율 집계 검증"""
    records = [
        {"em": True, "em_multi": True, "exec_match": True},
        {"em": False, "em_multi": True, "exec_match": True},
        {"em": False, "em_multi": False, "exec_match": False},
        {"em": False, "em_multi": False, "exec_match": True},
    ]
    s = scoring.summarize(records)
    assert s["n"] == 4 and s["em"] == 0.25 and s["em_multi"] == 0.5 and s["exec_acc"] == 0.75
    assert s["em_ci95"][0] <= 0.25 <= s["em_ci95"][1]


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
            question, _ = generate.split_example(ex, TOK)
            target = TOK.encode(preds[question]) + [EOS_ID]
            logits[b, -1, target[idx.size(1) - ex["sql_start"]]] = 1.0
        return logits
    model.forward = forward

    result = scoring.evaluate_set(model, "indist", TOK, torch.device("cpu"), None,
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
        scoring.evaluate_set(tiny_model(), "indist", TOK, torch.device("cpu"), None, eval_dir=tmp_path)


@pytest.mark.unit
def test_load_rules_provides_stage_interface():
    """현재 단계의 SQL 규칙 모듈이 evaluation·predict 가 쓰는 이름을 모두 제공해야 함"""
    rules = scoring.load_rules(config.STAGE)
    assert rules.STAGE == config.STAGE
    for name in ("parse_sql", "error_parts", "select_ambiguous", "multi_match", "MULTI_NOTE"):
        assert hasattr(rules, name), name


def test_label_mismatch_follows_label_rules():
    """규칙 일치 EM 의 제외 판정: 라벨 규칙(L2~L4)과 다른 정답만 True"""
    rules = scoring.load_rules(1)
    # L3: ID 를 요구하지 않는데 자기 ID 를 SELECT
    assert rules.label_mismatch("what item costs 135.87?", "SELECT item_id FROM items WHERE price = 135.87")
    assert not rules.label_mismatch("what is the item id of the thing that costs 135.87?",
                                    "SELECT item_id FROM items WHERE price = 135.87")
    # L4: 주문 단서 없이 orders
    assert rules.label_mismatch("run down everything tied to item 3839 for me",
                                "SELECT * FROM orders WHERE item_id = 3839")
    assert not rules.label_mismatch("which item costs 135.87?", "SELECT item_name FROM items WHERE price = 135.87")
