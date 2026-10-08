import json
from pathlib import Path

import pytest

from src import config
from src.evaluation import eval_sets


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
    assert eval_sets.list_sets(tmp_path) == ["a", "b"]
    assert eval_sets.list_sets(tmp_path / "missing") == []


@pytest.mark.unit
def test_write_sql_meta_keeps_used_sql_in_source_order(tmp_path):
    """sql.json 은 쌍에 쓰인 SQL 만, 원래 SQL 파일 순서대로 담는다"""
    _write(tmp_path / "x" / "pairs.json", PAIRS)
    _write(tmp_path / "sql.json", SQL_ROWS)
    eval_sets.write_sql_meta("x", tmp_path / "sql.json", tmp_path)
    pairs, meta = eval_sets.load_eval_set("x", tmp_path)
    assert pairs == PAIRS
    assert list(meta) == ["S2", "S1"]


@pytest.mark.unit
def test_build_sql_meta_missing_sql_raises(tmp_path):
    """쌍의 SQL 이 SQL 파일에 없으면 오류"""
    _write(tmp_path / "pairs.json", PAIRS)
    _write(tmp_path / "sql.json", SQL_ROWS[:2])
    with pytest.raises(ValueError):
        eval_sets.build_sql_meta(tmp_path / "pairs.json", tmp_path / "sql.json")


@pytest.mark.unit
def test_load_eval_set_stale_sql_meta_raises(tmp_path):
    """sql.json 을 만든 뒤 pairs.json 에 새 SQL 이 들어오면 불러올 때 오류"""
    _write(tmp_path / "x" / "pairs.json", PAIRS)
    _write(tmp_path / "x" / "sql.json", SQL_ROWS[2:])
    with pytest.raises(ValueError):
        eval_sets.load_eval_set("x", tmp_path)


@pytest.mark.unit
def test_committed_eval_sets_are_consistent():
    """저장소의 평가셋 폴더마다 sql.json 이 pairs.json 의 SQL 을 모두 담고 있어야 함"""
    names = eval_sets.list_sets()
    assert {"indist", "holdout"} <= set(names)
    for name in names:
        pairs, meta = eval_sets.load_eval_set(name)
        assert len(meta) == len({p["sql"] for p in pairs})


@pytest.mark.unit
def test_eval_sets_outside_train_dir():
    """평가셋 폴더는 학습 폴더(= 학습 데이터·BPE 코퍼스) 밖에 있어야 함"""
    train = config.TRAIN_DIR.resolve()
    assert train not in config.EVAL_DIR.resolve().parents and train != config.EVAL_DIR.resolve()


@pytest.mark.unit
def test_default_out_dir_stays_under_runs():
    """평가 결과는 커밋 폴더(models/)에 쓰지 않는다: runs/ 안 체크포인트는 그 폴더, 그 밖은 RELEASE_EVAL_DIR"""
    run_ckpt = config.RUNS_DIR / "some_run" / "best.pt"
    assert eval_sets.default_out_dir(run_ckpt) == run_ckpt.parent
    assert eval_sets.default_out_dir(config.MODEL_PATH) == config.RELEASE_EVAL_DIR
    assert eval_sets.default_out_dir("runs/stage1_old_run/best.pt") == Path("runs/stage1_old_run")
