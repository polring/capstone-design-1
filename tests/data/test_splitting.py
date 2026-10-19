"""SQL/contrast-group split isolation and deterministic assignment."""
import pytest

from src.data.splitting import split_by_sql, validate_split


def make_pairs(n_sql: int = 20, per_sql: int = 5) -> list[dict]:
    return [{"question": f"q{i}-{j}", "sql": f"SELECT * FROM items WHERE item_id = {3000 + i}"}
            for i in range(n_sql) for j in range(per_sql)]


# ===========================================================================
# split_by_sql
# ===========================================================================

@pytest.mark.unit
def test_split_by_sql_no_sql_overlap():
    """같은 SQL 의 질문들은 train / val 한쪽에만 들어가야 함 (val 이 암기 점수가 되지 않게)"""
    pairs = make_pairs()
    train, val = split_by_sql(pairs, val_ratio=0.2, seed=0)
    assert not {p["sql"] for p in train} & {p["sql"] for p in val}
    assert len({p["sql"] for p in val}) == 4  # 20 SQL * 0.2
    assert len(train) + len(val) == len(pairs)


@pytest.mark.unit
def test_split_by_sql_deterministic():
    """같은 seed 면 같은 분할, 다른 seed 면 다른 분할"""
    pairs = make_pairs()
    assert split_by_sql(pairs, 0.2, 0) == split_by_sql(pairs, 0.2, 0)
    assert split_by_sql(pairs, 0.2, 0)[1] != split_by_sql(pairs, 0.2, 1)[1]


@pytest.mark.unit
def test_split_by_sql_uses_original_pair_for_swapped_data():
    """이름 교체 쌍은 orig_sql 로 나누고, val 에는 원래 질문·SQL 이 들어가야 함 (기준 실행과 같은 val)"""
    pairs = make_pairs()
    swapped = [dict(p, question=p["question"] + " zorb", sql=p["sql"] + "9",
                    orig_question=p["question"], orig_sql=p["sql"]) if i % 2 else p
               for i, p in enumerate(pairs)]
    _, val_plain = split_by_sql(pairs, 0.2, 0)
    train_sw, val_sw = split_by_sql(swapped, 0.2, 0)
    assert val_sw == [{"question": p["question"], "sql": p["sql"]} for p in val_plain]
    assert all(p.get("orig_sql", p["sql"]) not in {v["sql"] for v in val_sw} for p in train_sw)


@pytest.mark.unit
def test_split_by_sql_groups_by_sql_group():
    """같은 sql_group(OR 값 순서만 다른 표기)은 한쪽에만 들어가고, val 의 정답은 그 쌍의 SQL 표기 그대로다"""
    pairs = [{"question": f"q{i}{k}", "sql": f"S{i} {'ab' if k else 'ba'}", "sql_group": f"S{i} ab"}
             for i in range(50) for k in range(2)]
    train, val = split_by_sql(pairs, 0.2, 0)
    val_groups = {p["sql_group"] for p in pairs if p["question"] in {v["question"] for v in val}}
    assert val and not val_groups & {p["sql_group"] for p in train}
    assert {v["sql"] for v in val} == {p["sql"] for p in pairs if p["sql_group"] in val_groups}


@pytest.mark.unit
def test_split_keeps_contrast_groups_and_unannotated_sql_together():
    """대비 네 SQL과 다른 파일에서 온 같은 SQL도 학습·검증으로 갈라지지 않는다"""
    pairs = [{"question": f"q{g}-{s}-{j}", "sql": f"S{g}-{s}", "sql_group": f"S{g}-{s}",
              "contrast_group": f"g{g}", "round": "scope"}
             for g in range(20) for s in range(4) for j in range(5)]
    pairs += [{"question": f"other{g}", "sql": f"S{g}-0"} for g in range(20)]
    train, val = split_by_sql(pairs, 0.2, 0)
    val_questions = {p["question"] for p in val}
    original_val = [p for p in pairs if p["question"] in val_questions]
    assert len(val) == 4 * 21
    validate_split(train, original_val)
    for g in range(20):
        assert len({p["question"] in val_questions for p in pairs
                    if p.get("contrast_group") == f"g{g}" or p["question"] == f"other{g}"}) == 1
    assert split_by_sql(list(reversed(pairs)), 0.2, 0)[1] == list(reversed(val))


@pytest.mark.unit
def test_explicit_split_rejects_transitive_contrast_leakage():
    """다른 대비 묶음이 공유 SQL로 연결돼도 명시적 분할의 누출을 거부한다"""
    a = {"question": "a", "sql": "A", "round": "scope", "contrast_group": "g1"}
    bridge1 = {"question": "bridge1", "sql": "B", "round": "scope", "contrast_group": "g1"}
    bridge2 = {"question": "bridge2", "sql": "B", "round": "scope", "contrast_group": "g2"}
    c = {"question": "c", "sql": "C", "round": "scope", "contrast_group": "g2"}
    with pytest.raises(ValueError, match="contrast group"):
        validate_split([a, bridge1, bridge2], [c])
    with pytest.raises(ValueError, match="share SQL"):
        validate_split([a], [{"question": "unannotated", "sql": "A"}])
    validate_split([a, bridge1, bridge2, c], [{"question": "other", "sql": "D"}])


@pytest.mark.unit
def test_contrast_group_names_are_scoped_by_round():
    """다른 라운드에서 같은 묶음 이름을 재사용해도 별개 SQL이면 분할을 허용한다"""
    validate_split([{"question": "a", "sql": "A", "round": "r1", "contrast_group": "001"}],
                   [{"question": "b", "sql": "B", "round": "r2", "contrast_group": "001"}])


@pytest.mark.unit
def test_split_without_contrasts_preserves_previous_seed_assignment():
    """대비 묶음이 없는 기존 데이터는 이전 SQL별 분할과 정확히 같다"""
    import random
    pairs = make_pairs()
    sqls = sorted({p["sql"] for p in pairs})
    random.Random(3).shuffle(sqls)
    held = set(sqls[:4])
    train, val = split_by_sql(pairs, 0.2, 3)
    assert train == [p for p in pairs if p["sql"] not in held]
    assert val == [p for p in pairs if p["sql"] in held]


