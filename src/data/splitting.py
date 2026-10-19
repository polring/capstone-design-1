# 역할: 같은 SQL과 대비 묶음이 분리되지 않도록 학습·검증 분할을 생성하고 검사한다.
# 사용처: train.py에서 검증 데이터 누수를 막고 재현 가능한 분할을 만들 때 사용한다.

"""Deterministic SQL/contrast-group train-validation splitting."""
import random


def split_keys(pairs: list[dict]) -> list[str]:
    """SQL 정규형과 라운드별 대비 묶음으로 연결된 SQL은 같은 분할 키를 쓴다."""
    sqls = [p.get("sql_group", p.get("orig_sql", p["sql"])) for p in pairs]
    parent = {sql: sql for sql in sqls}

    def find(sql):
        while parent[sql] != sql:
            parent[sql] = parent[parent[sql]]
            sql = parent[sql]
        return sql

    groups = {}
    for pair, sql in zip(pairs, sqls):
        if not pair.get("contrast_group"):
            continue
        group = (pair.get("round", ""), pair["contrast_group"])
        first = groups.setdefault(group, sql)
        a, b = find(first), find(sql)
        # 대표 SQL을 고정해 입력 파일·문장 순서가 바뀌어도 같은 seed 분할을 유지한다.
        parent[max(a, b)] = min(a, b)
    return [find(sql) for sql in sqls]


def validate_split(train: list[dict], val: list[dict]) -> None:
    """명시적으로 나눈 파일도 SQL·대비 묶음이 양쪽에 걸치면 거부한다."""
    keys = split_keys(train + val)
    if set(keys[:len(train)]) & set(keys[len(train):]):
        raise ValueError("train/validation share SQL or a contrast group")


def split_by_sql(pairs: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """SQL·대비 묶음 단위로 train/val 을 나눈다 (같은 묶음은 한쪽에만 들어간다).

    이름 교체 데이터(name_swap.py)는 바꾸기 전 SQL(orig_sql)을 기준으로 나누고, 검증에는 원래 쌍을 쓴다.
    그래서 어떤 학습 파일을 쓰든 검증셋은 원본 데이터 기준 실행과 같다.
    2단계 조합기 쌍은 OR 값을 질문에 나온 순서로 쓰므로 같은 SQL 이 여러 표기로 나온다. 이런 쌍은 sql_group
    (정규형 SQL)으로 묶어 나눈다. contrast_group 이 있으면 같은 round 의 대비 SQL도 함께 둔다.
    묶음 정보가 없는 다른 파일의 같은 SQL과 여러 대비 묶음이 공유하는 SQL도 함께 묶는다.
    val_ratio 는 연결된 묶음 수 기준이므로 실제 문항 비율은 다를 수 있다.
    """
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be within 0..1")
    keys = split_keys(pairs)
    sqls = sorted(set(keys))
    random.Random(seed).shuffle(sqls)
    val_sqls = set(sqls[: int(len(sqls) * val_ratio)])
    train = [p for p, key in zip(pairs, keys) if key not in val_sqls]
    val = [{"question": p.get("orig_question", p["question"]), "sql": p.get("orig_sql", p["sql"])}
           for p, key in zip(pairs, keys) if key in val_sqls]
    return train, val


