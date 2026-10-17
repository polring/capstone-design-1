import json
import os

import pytest

from src.config import stage_paths
from src.data.stage2 import sql_gen as g
from src.data.stage2.sql_ast import violations

DB_DIR = str(stage_paths(1).db_dir)
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not os.path.exists(os.path.join(DB_DIR, "shop.db")),
                                 reason="1단계 shop.db 없음 (db_gen build 먼저)")]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("s2")
    # 유일 키 = 칸(5개)은 1단계 모양으로만 채워져서, 작은 설정에서는 1단계 모양 상한을 넉넉히 둔다
    cfg = g.GenConfig(n_train=1500, n_eval_indist=150, n_eval_struct=150, cell_floor=30, cell_floor_train=20,
                      value_floors={"items.stock = 0": 40, "items.stock > 0": 40}, value_floor_train=30,
                      stage1_ratio_max=0.2, exists_floor=40, exists_floor_train=30)
    g.build(DB_DIR, str(out), cfg)
    return out


def test_verify_passes(built):
    """작은 설정으로 만든 결과가 검증(결정성 포함)을 통과한다"""
    assert g.verify(str(built)) == []


def test_like_fragments_never_whole_name():
    """LIKE 조각은 어떤 이름 전체(미등장 이름 포함)와도 같지 않다"""
    con, ho, bans = g._open(DB_DIR)
    pools = g.Pools(con, bans, g.db_gen.Config())
    for (t, c), modes in pools.like.items():
        names = set(pools.values[(t, c)]) | bans[t][c]
        for frags in modes.values():
            assert frags and not set(frags) & names
    con.close()


def test_struct_combos_absent_from_train(built):
    """구조 홀드아웃 조합은 학습·분포 내 평가에 없다"""
    combos = g.combos_of(g.GenConfig())
    for fn in (g.FILES["train"], g.FILES["indist"]):
        for d in json.load(open(built / fn, encoding="utf-8")):
            assert g.struct_hit(g._query(d), combos) is None


def test_holdout_uses_banned_literal(built):
    """미등장 평가 SQL 은 모두 홀드아웃 리터럴을 포함하고 규칙을 지킨다"""
    _, _, bans = g._open(DB_DIR)
    rows = json.load(open(built / g.FILES["holdout"], encoding="utf-8"))
    assert rows
    for d in rows:
        q = g._query(d)
        assert violations(q) == []
        assert any(set(c.val if c.op == "IN" else [c.val]) & bans[q.table].get(c.col, set())
                   for c in q.conds if c.op in ("=", "IN"))
        assert d["holdout_shape"] in g.HOLDOUT_SHAPES


def test_no_stage1_overlap(built):
    """1단계 평가 SQL 은 학습에 없고, 1단계 학습 SQL 은 평가 분할에 없다"""
    s1_train, s1_eval = g.stage1_sqls(g.GenConfig().stage1_data_dir)
    assert s1_train and s1_eval
    sqls = {fn: {d["sql"] for d in json.load(open(built / fn, encoding="utf-8"))} for fn in g.FILES.values()}
    assert not sqls[g.FILES["train"]] & s1_eval
    for fn in (g.FILES["indist"], g.FILES["struct"], g.FILES["holdout"]):
        assert not sqls[fn] & s1_train
