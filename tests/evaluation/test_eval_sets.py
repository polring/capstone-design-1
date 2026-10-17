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
def test_sql_group_keys_meta_and_several_sql_files(tmp_path):
    """쌍에 sql_group(정규형)이 있으면 메타데이터를 그것으로 찾는다. SQL 파일은 여러 개를 줄 수 있다"""
    pairs = [{"question": "q1", "sql": "S1 (b, a)", "sql_group": "S1 (a, b)"}, {"question": "q2", "sql": "S2"}]
    _write(tmp_path / "x" / "pairs.json", pairs)
    _write(tmp_path / "s1.json", [{"sql": "S1 (a, b)", "table": "t"}])
    _write(tmp_path / "s2.json", [{"sql": "S2", "table": "t"}, {"sql": "S1 (a, b)", "table": "dup"}])
    eval_sets.write_sql_meta("x", [tmp_path / "s1.json", tmp_path / "s2.json"], tmp_path)
    _, meta = eval_sets.load_eval_set("x", tmp_path)
    assert list(meta) == ["S1 (a, b)", "S2"] and meta["S1 (a, b)"]["table"] == "t"


@pytest.mark.unit
def test_combined_file_splits_by_set_and_replaces_only_given_sets(tmp_path):
    """합친 방식: pairs.json 하나의 set 으로 평가셋을 나누고, 메타데이터는 그 평가셋의 SQL 만 준다.
    write_combined 는 준 평가셋만 바꾸고 나머지는 두며, drop 은 지운다. 폴더 방식 평가셋과 함께 읽힌다"""
    _write(tmp_path / "sql_src.json", SQL_ROWS)
    eval_dir = tmp_path / "eval"
    eval_sets.write_combined({"b": PAIRS[:2], "a": PAIRS[2:]}, tmp_path / "sql_src.json", eval_dir)
    _write(eval_dir / "folder" / "pairs.json", PAIRS)
    eval_sets.write_sql_meta("folder", tmp_path / "sql_src.json", eval_dir)
    assert eval_sets.list_sets(eval_dir) == ["a", "b", "folder"]
    pairs, meta = eval_sets.load_eval_set("b", eval_dir)
    assert pairs == PAIRS[:2] and list(meta) == ["S1"]  # set 필드는 빼고 준다
    eval_sets.write_combined({"a": PAIRS[:1]}, tmp_path / "sql_src.json", eval_dir)
    assert eval_sets.load_eval_set("a", eval_dir)[0] == PAIRS[:1]
    assert eval_sets.load_eval_set("b", eval_dir)[0] == PAIRS[:2]
    eval_sets.write_combined({}, tmp_path / "sql_src.json", eval_dir, drop={"b"})
    assert eval_sets.list_sets(eval_dir) == ["a", "folder"]
    assert [r["sql"] for r in json.loads((eval_dir / "sql.json").read_text(encoding="utf-8"))] == ["S1"]
    with pytest.raises(FileNotFoundError):
        eval_sets.load_eval_set("b", eval_dir)


@pytest.mark.unit
def test_committed_eval_sets_are_consistent():
    """저장소의 평가셋 폴더마다 sql.json 이 pairs.json 의 SQL 을 모두 담고 있어야 함"""
    names = eval_sets.list_sets()
    assert {"indist", "holdout"} <= set(names)
    for name in names:
        pairs, meta = eval_sets.load_eval_set(name)
        assert len(meta) == len({eval_sets.sql_key(p) for p in pairs})


@pytest.mark.integration
def test_stage2_eval_sets_consistent_and_unseen_in_training():
    """2단계 평가셋: sql.json 이 쌍의 SQL 을 모두 담고, 평가 SQL 은 학습 SQL(sql_gen train.json)에 없어야 함"""
    paths = config.stage_paths(2)
    names = eval_sets.list_sets(paths.eval_dir)
    assert {"indist", "struct", "holdout", "phrase"} <= set(names)
    train = {r["sql"] for r in json.loads((paths.sql_dir / "train.json").read_text(encoding="utf-8"))}
    assert paths.train_dir.resolve() not in paths.eval_dir.resolve().parents
    for name in names:
        pairs, meta = eval_sets.load_eval_set(name, paths.eval_dir)
        assert not set(meta) & train, name
        assert all(p["question"] == p["question"].lower() for p in pairs), name  # 추론처럼 소문자


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
