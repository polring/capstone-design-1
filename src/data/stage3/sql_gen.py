# 역할: 3단계 문법과 값·구조 분할 정책에 따라 정답 SQL을 생성하고 검사한다.
# 사용처: 3단계 학습·평가 질문을 만들기 전에 SQL과 분할 메타데이터를 준비할 때 사용한다.

"""
Text-to-SQL 3단계 SQL 생성기

2단계 생성기를 복사해 3단계 DB, 두 단어 이름, 날짜 비교·정렬·holdout을 지원한다.
build / verify는 일반 조회·날짜·집계·GROUP BY·HAVING·스칼라 서브쿼리를 함께 처리한다.

절차
  1. 커버리지 하한: (테이블, 컬럼, 연산자) 칸마다 cell_floor 개를 그 칸을 포함하도록 강제 샘플링
     고정값 표현 하한: value_floors 의 조건(stock = 0 / stock > 0, 라벨 규칙 L16)을 포함한 SQL 을 정한 개수만큼
     존재 확인 하한: L1(SELECT = 조건 컬럼, 조건 하나 =) SQL 을 exists_floor 개 (1단계에서 존재 확인 질문이 적어 틀렸다)
  2. 목표 분포: 조건 수 / IN / AND·OR / ORDER BY / LIMIT 비중에 따라 나머지를 샘플링
     - IN 은 같은 컬럼의 값 2~3개 나열(조건 하나), OR 는 서로 다른 컬럼 조건끼리
     - 상한(caps): ID 조건 포함 SQL, 결과 0행 SQL(없는 ID 제외), 1단계 모양 SQL(조건 1개 = , 정렬 없음, 존재 확인 제외)
     - 빈 결과·전체 행·1행 결과도 유효하다. 비율 상한은 분포 설정이며 SQL 유효성 규칙이 아니다.
  3. 분할
     - struct : struct_holdout 조합에 걸리는 SQL. 학습·분포 내 평가에 절대 넣지 않음 (구조 일반화 평가)
     - train / indist : 나머지를 SQL 단위로 분할
     - holdout : 개체·ID 및 날짜 holdout 리터럴을 넣은 SQL (날짜는 비교·BETWEEN도 포함)
  3-2. 조합 하한: 학습 SQL 에서 구조 특징 두 개의 쌍(예: 테이블×정렬 방향, 조건 종류×조건 종류, 함수×그룹 키)마다
     pair_floor 개를 채운다. 모자란 쌍부터 그 쌍을 겨냥해 샘플링하고, 모든 쌍이 하한보다 많은 학습 SQL 과 바꾼다(학습 수 유지).
     기존 하한·상한·holdout·구조 홀드아웃은 그대로 지킨다. 샘플링으로 채우지 못한 쌍은 리포트에 남긴다
  4. 1단계와 겹침 금지 (1단계 쌍을 2단계 학습에 섞고, 1단계 평가셋 점수도 계속 재기 때문)
     - 1단계 평가 SQL 은 학습·분포 내 평가에 넣지 않음
     - 1단계 학습 SQL 은 2단계 평가 분할(indist / struct / holdout)에 넣지 않음

사용법 (저장소 루트에서)
    python -m src.data.stage3.sql_gen build
    python -m src.data.stage3.sql_gen verify
    python -m src.data.stage3.sql_gen summary
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from src.config import stage_paths
from src.data import dataset
from src.data import db_gen
from src.evaluation import eval_sets
from src.data.stage3 import schema, sql_ast
from src.data.stage3.sql_ast import Cond, Query, canonical, features, from_dict, parse, skeleton, to_dict, violations

PRICE_ROUND = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150]
LIMIT_WEIGHTS = {1: 35, 2: 6, 3: 15, 4: 3, 5: 15, 6: 2, 7: 2, 8: 2, 9: 2, 10: 15}


def _default_struct_holdout() -> list[list[list]]:
    # 사용자가 드물게 쓰는 조합만 뺀다. 자주 쓰는 조합(값 나열 × ORDER BY, orders 의 BETWEEN·!=)은 학습에 둔다.
    # LIKE × != 는 items 만: customers 의 != 는 대부분 LIKE 와 붙어서, 빼면 city·membership != 칸이 얇아진다
    return [
        [["n", 3], ["limit", True]],
        [["op", "LIKE"], ["order", True]],
        [["op", "LIKE"], ["op", "BETWEEN"]],
        [["op", "LIKE"], ["op", "!="], ["table", "items"]],
        [["op", "BETWEEN"], ["op", "!="]],
    ]


@dataclass
class GenConfig:
    seed: int = 0
    n_train: int = 40000
    n_eval_indist: int = 2000
    n_eval_struct: int = 2000
    # 미등장 값 평가: 리터럴마다 만들 수 있는 모양을 전부 하나씩 (만들 수 없는 모양은 건너뜀)
    holdout_shapes_unique: list = field(default_factory=lambda: ["single", "in2", "in3"])
    holdout_shapes_fk: list = field(default_factory=lambda: [
        "single", "in2", "in3", "and2", "and3", "order", "order_limit", "and2_order", "in2_and"])
    # 목표 분포
    n_cond_weights: dict = field(default_factory=lambda: {0: 5, 1: 35, 2: 40, 3: 20})
    table_weights: dict = field(default_factory=lambda: {"customers": 1.0, "items": 1.0, "orders": 1.6})
    in_ratio: float = 0.10              # IN 을 쓸 수 있는 컬럼의 조건을 IN 으로 만드는 비율
    # 유일 키(이름·PK)의 IN 샘플링 비율. 단독/복합 조건 모두 유효하다.
    in_ratio_unique: float = 0.5
    in3_ratio: float = 0.4              # IN 값 3개 비율 (나머지는 2개)
    or_ratio: float = 0.04              # 조건 2개 이상인 SQL 중 서로 다른 컬럼 OR 비율
    order_ratio: float = 0.50           # 실제 비중은 '의미 없는 정렬' 필터 때문에 약 30%
    limit_ratio: float = 0.85           # 실제 비중은 정렬 SQL 의 약 80%
    id_col_weight: float = 0.5          # 조건 컬럼을 고를 때 ID 컬럼 가중치 (나머지 1)
    id_nonexistent_ratio: float = 0.20  # ID = 리터럴 중 존재하지 않는 ID 비율 (1단계와 같음)
    price_int_ratio: float = 0.50       # price 범위 조건 리터럴 중 정수 어림수 비율
    # 상한 (학습 + 분포 내 평가 기준)
    id_ratio_max: float = 0.30
    empty_ratio_max: float = 0.10
    stage1_ratio_max: float = 0.05
    # 커버리지 하한
    cell_floor: int = 200        # 생성 시 칸마다 강제로 채우는 개수
    cell_floor_train: int = 150  # verify: 분할 후 학습에 남아야 하는 개수
    struct_holdout: list = field(default_factory=_default_struct_holdout)
    stage1_data_dir: str = str(stage_paths(1).data_dir)  # 겹침 금지용 1단계 쌍 (train/*.json, eval/<이름>/pairs.json)
    # 고정값 표현 하한 (라벨 규칙 L16: out of stock = stock = 0, in stock = stock > 0). 무작위로는 각 40개 안팎만 나온다
    value_floors: dict = field(default_factory=lambda: {"items.stock = 0": 1000, "items.stock > 0": 1000})
    value_floor_train: int = 800  # verify: 분할 후 학습에 남아야 하는 개수
    # 존재 확인(L1) 하한. 무작위로는 360개 안팎(0.9%)만 나온다. 1단계 모양이지만 상한에서 뺀다
    exists_floor: int = 1000
    n_aggregate_train: int = 60000
    n_aggregate_eval: int = 1000
    exists_floor_train: int = 850  # verify: 분할 후 학습에 남아야 하는 개수
    # 조합 하한: 학습 SQL 의 구조 특징 쌍마다 최소 개수 (일반 조회·집계 계열 각각). 0 이면 끄기
    pair_floor: int = 50

    def __post_init__(self):
        # 리포트 JSON 에서 다시 읽으면 dict 키가 문자열이 된다
        self.n_cond_weights = {int(k): v for k, v in self.n_cond_weights.items()}


# ---------------------------------------------------------------------------
# 리터럴 풀
# ---------------------------------------------------------------------------

class Pools:
    def __init__(self, con: sqlite3.Connection, bans: dict, dbcfg: db_gen.Config):
        self.con = con
        self.bans = bans
        self.values: dict[tuple[str, str], list] = {}
        self.nonexistent: dict[tuple[str, str], list[int]] = {}
        self.n_rows = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in schema.TABLE_COLUMNS}
        domain = {"customer_id": ("customers", dbcfg.customer_id_range),
                  "item_id": ("items", dbcfg.item_id_range),
                  "order_id": ("orders", dbcfg.order_id_range)}
        for t, cols in schema.TABLE_COLUMNS.items():
            for c in cols:
                banned = bans.get(t, {}).get(c, set())
                rows = {r[0] for r in con.execute(f"SELECT DISTINCT {c} FROM {t}")}
                self.values[(t, c)] = sorted(rows - banned)
                if schema.kind(t, c) == "id":
                    dt, (lo, hi) = domain[c]
                    existing = {r[0] for r in con.execute(f"SELECT {c} FROM {dt}")}
                    self.nonexistent[(t, c)] = [v for v in range(lo, hi + 1)
                                                if v not in existing and v not in banned]

        # LIKE 조각: 학습에 쓸 수 있는 이름에서 자른다. 어떤 이름 전체(미등장 이름 포함)와 같은 조각은 쓰지 않는다
        self.like: dict[tuple[str, str], dict[str, list[str]]] = {}
        for t, c in (("customers", "name"), ("items", "item_name")):
            all_names = {r[0] for r in con.execute(f"SELECT {c} FROM {t}")} | bans[t][c]
            frags: dict[str, set[str]] = {"prefix": set(), "suffix": set(), "contains": set()}
            for nm in self.values[(t, c)]:
                for k in (1, 2):
                    frags["prefix"].add(nm[:k])
                for k in (2, 3):
                    frags["suffix"].add(nm[-k:])
                    frags["contains"] |= {nm[i:i + k] for i in range(1, len(nm) - k)}  # 처음·끝이 아닌 위치
            self.like[(t, c)] = {m: sorted(f for f in s if f.isascii() and f.isalpha() and f not in all_names) for m, s in frags.items()}

        self._cond_count: dict[str, int] = {}

    def cond_count(self, table: str, cond: Cond) -> int:
        sql = f"SELECT COUNT(*) FROM {table} WHERE {cond.render(table)}"
        if sql not in self._cond_count:
            self._cond_count[sql] = self.con.execute(sql).fetchone()[0]
        return self._cond_count[sql]


def _threshold(rng: random.Random, table: str, col: str, cfg: GenConfig):
    """범위 조건 리터럴. price 는 정수 어림수와 소수를 섞어, 소수점 유무로 컬럼을 맞히지 못하게 한다."""
    if (table, col) == ("items", "price"):
        if rng.random() < cfg.price_int_ratio:
            return float(rng.choice(PRICE_ROUND))
        if rng.random() < 0.4:
            return rng.randint(1, 198) + 0.99
        return round(rng.uniform(1.0, 199.99), 2)
    if col == "stock":
        return rng.randrange(0, 101, 5) if rng.random() < 0.5 else rng.randint(0, 100)
    if col == "quantity":
        return rng.randint(1, 10)
    raise ValueError(f"범위 조건 불가 컬럼 {table}.{col}")


def _eq_literal(rng: random.Random, pools: Pools, table: str, col: str, cfg: GenConfig) -> tuple[object, bool]:
    """= 리터럴과 '존재하지 않는 ID' 여부."""
    if schema.kind(table, col) == "id" and rng.random() < cfg.id_nonexistent_ratio:
        return rng.choice(pools.nonexistent[(table, col)]), True
    return rng.choice(pools.values[(table, col)]), False


def _make_in(rng: random.Random, pools: Pools, table: str, col: str, cfg: GenConfig,
             k: int | None = None, include=None) -> tuple[Cond | None, bool]:
    """IN 조건: 서로 다른 값 k 개 (k 없으면 2 또는 3). include 가 있으면 그 값을 넣는다 (미등장 값 평가)."""
    k = k or (3 if rng.random() < cfg.in3_ratio else 2)
    vals, nonexist = ([include] if include is not None else []), False
    for _ in range(k * 10):
        if len(vals) == k:
            break
        v, ne = _eq_literal(rng, pools, table, col, cfg)
        if v not in vals:
            vals.append(v)
            nonexist |= ne
    if len(vals) != k:
        return None, False
    rng.shuffle(vals)
    return Cond(col, "IN", tuple(vals)), nonexist


def _pick_op(rng: random.Random, table: str, col: str, conn: str | None, cfg: GenConfig) -> str:
    """지원되는 행 조건 선택. 연결어에 따른 이름·ID·부정 금지는 없다."""
    ratio = cfg.in_ratio_unique if (table, col) in schema.UNIQUE_KEYS else cfg.in_ratio
    if schema.in_ok(table, col) and rng.random() < ratio:
        return "IN"
    return rng.choice(schema.OPS[schema.kind(table, col)])


def _make_cond(rng: random.Random, pools: Pools, table: str, col: str, op: str, cfg: GenConfig
               ) -> tuple[Cond | None, bool]:
    if schema.kind(table, col) == "date":
        values = pools.values[(table, col)]
        if op == "BETWEEN":
            lo, hi = sorted(rng.sample(values, 2))
            return Cond(col, op, lo, hi), False
        return Cond(col, op, rng.choice(values)), False
    if op == "IN":
        return _make_in(rng, pools, table, col, cfg)
    if op == "=":
        val, nonexist = _eq_literal(rng, pools, table, col, cfg)
        return Cond(col, op, val), nonexist
    if op == "!=":
        return Cond(col, op, rng.choice(pools.values[(table, col)])), False
    if op == "LIKE":
        mode = rng.choice(["prefix", "suffix", "contains"])
        frag = rng.choice(pools.like[(table, col)][mode])
        pat = {"prefix": f"{frag}%", "suffix": f"%{frag}", "contains": f"%{frag}%"}[mode]
        return Cond(col, op, pat), False
    if op == "BETWEEN":
        a, b = _threshold(rng, table, col, cfg), _threshold(rng, table, col, cfg)
        lo, hi = min(a, b), max(a, b)
        return Cond(col, op, lo, hi), False
    return Cond(col, op, _threshold(rng, table, col, cfg)), False


def _pick_cols(rng: random.Random, table: str, k: int, exclude: set[str], cfg: GenConfig,
               no_id: bool = False) -> list[str]:
    cands = [c for c in schema.TABLE_COLUMNS[table]
             if c not in exclude and not (no_id and schema.kind(table, c) == "id")]
    out: list[str] = []
    while cands and len(out) < k:
        w = [cfg.id_col_weight if schema.kind(table, c) == "id" else 1.0 for c in cands]
        c = rng.choices(cands, weights=w)[0]
        out.append(c)
        cands.remove(c)
    return out


# ---------------------------------------------------------------------------
# SQL 하나 샘플링
# ---------------------------------------------------------------------------

def sample_query(rng: random.Random, pools: Pools, cfg: GenConfig,
                 cell: tuple[str, str, str] | None = None,
                 fixed: tuple[str, str, object] | None = None,
                 shape: dict | None = None,
                 pin: tuple[str, str, str, object] | None = None, exists: bool = False) -> Query | None:
    """규칙을 만족하는 Query 하나. 실패하면 None (호출 쪽에서 다시 시도).

    cell  = (table, col, op)   : 그 칸의 조건을 반드시 포함 (커버리지 하한용)
    fixed = (table, col, value): 그 값의 = 조건(shape 에 "in" 이 있으면 IN 조건)을 반드시 포함 (미등장 값 평가용)
    shape = HOLDOUT_SHAPES 의 값: 조건 수·연결어·IN 값 수·정렬·LIMIT 유무를 고정
    pin   = (table, col, op, value): 그 조건을 반드시 포함 (고정값 표현 하한용, 예: stock = 0)
    exists: 존재 확인(L1) SQL: 조건 하나 =, SELECT 는 그 조건 컬럼, 정렬 없음 (존재 확인 하한용)
    """
    tables = list(schema.TABLE_COLUMNS)
    forced = cell or fixed or pin
    table = forced[0] if forced else rng.choices(tables, weights=[cfg.table_weights[t] for t in tables])[0]
    cols = schema.TABLE_COLUMNS[table]

    weights = dict(cfg.n_cond_weights)
    if forced:
        weights.pop(0, None)
    n = rng.choices(list(weights), weights=list(weights.values()))[0]
    n = int(n)
    conn = None
    if n >= 2:
        conn = "OR" if rng.random() < cfg.or_ratio else "AND"
    if shape:
        n, conn = shape["n"], shape["conn"]
    if exists:
        n, conn = 1, None

    conds: list[Cond] = []
    nonexist = False
    if forced:
        first_col = forced[1]
        if fixed and shape and shape.get("in"):
            c, ne = _make_in(rng, pools, table, first_col, cfg, k=shape["in"], include=fixed[2])
        elif fixed:
            op = shape.get("date_op", "=") if shape else "="
            if op == "BETWEEN":
                other = rng.choice(pools.values[(table, first_col)])
                lo, hi = sorted((fixed[2], other))
                c = Cond(first_col, op, lo, hi)
            else:
                c = Cond(first_col, op, fixed[2])
            ne = False
        elif pin:
            c, ne = Cond(first_col, pin[2], pin[3]), False
        else:
            c, ne = _make_cond(rng, pools, table, first_col, cell[2], cfg)
        if c is None:
            return None
        conds.append(c)
        nonexist |= ne
        rest = _pick_cols(rng, table, n - 1, {first_col}, cfg)
    else:
        rest = _pick_cols(rng, table, n, set(), cfg)
    for col in rest:
        # 미등장 값 평가는 그 리터럴만 새 값이 되도록 나머지 조건에 IN 을 쓰지 않는다
        op = "=" if exists else _pick_op(rng, table, col, conn, cfg)
        if fixed and op == "IN":
            op = "="
        c, ne = _make_cond(rng, pools, table, col, op, cfg)
        if c is None:
            return None
        conds.append(c)
        nonexist |= ne

    # ORDER BY / LIMIT
    eq_cols = {c.col for c in conds if c.op in ("=", "IN")}
    order = limit = None
    want_order = False if exists else shape["order"] if shape else rng.random() < cfg.order_ratio
    if want_order:
        ocands = [c for c in cols if schema.orderable(table, c) and c not in eq_cols]
        if ocands:
            order = (rng.choice(ocands), rng.choice(["ASC", "DESC"]))
            if (shape["limit"] if shape else rng.random() < cfg.limit_ratio):
                limit = rng.choices(list(LIMIT_WEIGHTS), weights=list(LIMIT_WEIGHTS.values()))[0]
        elif shape:
            return None  # 정렬이 필요한데 정렬할 컬럼이 없음

    select = conds[0].col if exists else rng.choice(cols + ["*"])
    q = canonical(Query(table, select, tuple(conds), conn, order, limit))
    if violations(q):
        return None

    n_rows = len(pools.con.execute(q.render()).fetchall())
    meta = {"n_rows": n_rows, "is_nonexistent": nonexist,
            "has_id": any(schema.kind(table, c.col) == "id" for c in q.conds)}
    return replace(q, meta=meta)


# ---------------------------------------------------------------------------
# 분류 술어
# ---------------------------------------------------------------------------

def is_exists(q: Query) -> bool:
    """존재 확인(L1): 조건 하나 = 이고 SELECT 가 그 조건 컬럼."""
    return len(q.conds) == 1 and q.conds[0].op == "=" and q.select == q.conds[0].col


def is_stage1_shape(q: Query) -> bool:
    """1단계 모양(조건 1개 =, 정렬 없음). 존재 확인은 하한을 따로 두므로 여기서 뺀다."""
    return len(q.conds) == 1 and q.conds[0].op == "=" and q.order is None and not is_exists(q)


def is_empty(q: Query) -> bool:
    return q.meta.get("n_rows", 1) == 0 and not q.meta.get("is_nonexistent")


def combos_of(cfg: GenConfig) -> list[set[tuple]]:
    return [{tuple(a) for a in combo} for combo in cfg.struct_holdout]


def struct_hit(q: Query, combos: list[set[tuple]]) -> int | None:
    f = features(q)
    for i, combo in enumerate(combos):
        if combo <= f:
            return i
    return None


def cells() -> list[tuple[str, str, str]]:
    return [(t, c, op) for t, cols in schema.TABLE_COLUMNS.items() for c in cols for op in schema.ops(t, c)]


def cell_fully_held(cell: tuple[str, str, str], combos: list[set[tuple]]) -> bool:
    """그 칸의 SQL 이 전부 구조 홀드아웃으로 가는가 (예: orders 의 != )."""
    return any(combo <= {("table", cell[0]), ("op", cell[2])} for combo in combos)


# ---------------------------------------------------------------------------
# 조합 하한 (구조 특징 쌍)
# ---------------------------------------------------------------------------

MULTI_FEATURES = {"cond", "col", "icond"}  # 한 SQL 에 여러 개 올 수 있어 같은 종류끼리도 짝짓는 특징


def _op_class(op: str) -> str:
    return "cmp" if op in ("<", ">", "<=", ">=") else op


def _cond_features(table: str, conds, prefix: str = "cond") -> set[tuple[str, object]]:
    f: set[tuple[str, object]] = set()
    for c in conds:
        f.add((prefix, f"{schema.kind(table, c.col)}:{_op_class(c.op)}"))
        if prefix == "cond":
            f.add(("col", c.col))
    return f


def plain_features(q: Query) -> set[tuple[str, object]]:
    """일반 조회의 조합 하한 특징: 테이블, SELECT 역할, 조건 수·연결어, 조건 종류×연산자, 조건 컬럼, 정렬, LIMIT."""
    return {("tab", q.table), ("sel", schema.select_kind(q.table, q.select, q.cond_cols())), ("n", len(q.conds)),
            ("conn", q.conn or "-"), ("ord", f"{q.order[0]}:{q.order[1]}" if q.order else "-"),
            ("lim", sql_ast.limit_kind(q.limit))} | _cond_features(q.table, q.conds)


def feature_pairs(features) -> list[tuple]:
    fs = sorted(features, key=str)
    return [(a, b) for i, a in enumerate(fs) for b in fs[i + 1:] if a[0] != b[0] or a[0] in MULTI_FEATURES]


def pair_label(pair: tuple) -> str:
    return " | ".join(f"{k}={v}" for k, v in pair)


def short_pairs(queries, features_of, floor: int) -> dict[str, int]:
    """학습에 나온 특징 쌍 중 하한보다 적은 것 (라벨 → 개수)."""
    counts = Counter(p for q in queries for p in feature_pairs(features_of(q)))
    return {pair_label(p): n for p, n in sorted(counts.items(), key=lambda kv: (kv[1], str(kv[0]))) if n < floor}


def ensure_pair_floor(training: list, features_of, sample, floor: int, rng: random.Random,
                      removable=lambda old, new: True, on_swap=None, stratum=lambda q: None,
                      attempts_per_pair: int = 3000) -> dict:
    """특징 쌍마다 floor 개가 되도록 training 의 여분 SQL 을 새 SQL 로 바꾼다 (training 을 제자리에서 고침).

    모자란 쌍부터 sample(쌍)으로 후보를 뽑고, 모자란 쌍을 하나라도 채우는 후보만 받는다. 바꿀 자리는 모든 쌍이 floor 보다
    많은 학습 SQL 이다(빼도 어느 쌍도 하한 밑으로 내려가지 않음). removable(old, new)로 다른 하한·상한을 지킨다.
    같은 stratum(예: 집계 계열)의 SQL 과 먼저 바꿔 계열 비율을 유지하고, 없을 때만 다른 stratum 과 바꾼다.
    한 쌍에 attempts_per_pair 번 뽑아도 못 채우면 그 쌍은 포기하고 리포트에 남긴다.
    """
    pairs_of = [feature_pairs(features_of(q)) for q in training]
    counts: Counter = Counter(p for ps in pairs_of for p in ps)
    order = list(range(len(training)))
    rng.shuffle(order)
    groups: dict = {}
    for i in order:
        groups.setdefault(stratum(training[i]), []).append(i)
    cursors: Counter = Counter()
    given_up: set = set()
    swapped = 0
    full = False

    def scan(key, new):
        members = groups.get(key, [])
        for _ in range(len(members)):
            i = members[cursors[key] % len(members)]
            cursors[key] += 1
            if all(counts[p] > floor for p in pairs_of[i]) and removable(training[i], new):
                return i
        return None

    def slot(new):
        key = stratum(new)
        i = scan(key, new)
        for other in sorted(groups, key=str):
            if i is not None:
                break
            if other != key:
                i = scan(other, new)
                if i is not None:
                    groups[other].remove(i)
                    groups.setdefault(key, []).append(i)
        return i

    while not full:
        short = sorted((p for p, n in counts.items() if n < floor and p not in given_up), key=lambda p: (counts[p], str(p)))
        if not short:
            break
        target = short[0]
        for _ in range(attempts_per_pair):
            if counts[target] >= floor:
                break
            q = sample(target)
            if q is None:
                continue
            ps = feature_pairs(features_of(q))
            for p in ps:
                if p not in counts:
                    counts[p] = 0
            if not any(counts[p] < floor for p in ps):
                continue
            i = slot(q)
            if i is None:
                full = True
                break
            for p in pairs_of[i]:
                counts[p] -= 1
            for p in ps:
                counts[p] += 1
            if on_swap:
                on_swap(training[i], q)
            training[i], pairs_of[i] = q, ps
            swapped += 1
        if counts[target] < floor:
            given_up.add(target)
    short_now = {pair_label(p): n for p, n in sorted(counts.items(), key=lambda kv: (kv[1], str(kv[0]))) if n < floor}
    return {"floor": floor, "n_pairs": len(counts), "swapped": swapped, "budget_exhausted": full,
            "n_short": len(short_now), "short": short_now}


def _plain_hint(rng: random.Random, target: tuple) -> tuple[str, str, str] | None:
    """쌍을 겨냥한 칸 (테이블, 컬럼, 연산자). 조건 특징이 없거나 맞는 칸이 없으면 None (그냥 샘플링)."""
    feats = dict()
    for k, v in target:
        feats.setdefault(k, []).append(v)
    tables = [feats["tab"][0]] if "tab" in feats else list(schema.TABLE_COLUMNS)
    options = []
    for t in tables:
        for col in schema.TABLE_COLUMNS[t]:
            if "col" in feats and col not in feats["col"]:
                continue
            for op in schema.ops(t, col):
                kind_op = f"{schema.kind(t, col)}:{_op_class(op)}"
                if "cond" in feats and kind_op not in feats["cond"]:
                    continue
                options.append((t, col, op))
    if not options or not ({"col", "cond"} & set(feats)):
        return None
    return rng.choice(options)


# ---------------------------------------------------------------------------
# 전체 생성
# ---------------------------------------------------------------------------

def stage1_sqls(data_dir: str) -> tuple[set[str], set[str]]:
    """1단계 (학습 SQL, 평가 SQL). 학습은 train/*.json 전부, 평가는 eval/ 아래 평가셋 전부."""
    d = Path(data_dir)
    train = {p["sql"] for p in dataset.load_train_pairs(train_dir=d / "train")}
    ev = {p["sql"] for name in eval_sets.list_sets(d / "eval") for p in dataset.load_eval_pairs(name, d / "eval")}
    return train, ev


def parse_value_key(key: str) -> tuple[str, str, str, object]:
    """'items.stock = 0' → (table, col, op, value)."""
    tc, op, val = key.split(" ")
    table, col = tc.split(".")
    return table, col, op, float(val) if "." in val else int(val)


def _open(db_dir: str):
    con = sqlite3.connect(os.path.join(db_dir, "shop.db"))
    ho = json.load(open(os.path.join(db_dir, "holdout.json"), encoding="utf-8"))
    return con, ho, db_gen.load_bans(ho)


def generate(con: sqlite3.Connection, ho: dict, cfg: GenConfig) -> dict[str, list[Query]]:
    bans = db_gen.load_bans(ho)
    rng = random.Random(cfg.seed)
    pools = Pools(con, bans, db_gen.Config(**{k: tuple(v) if isinstance(v, list) else v
                                              for k, v in ho["config"].items()}))
    combos = combos_of(cfg)
    s1_train, s1_eval = stage1_sqls(cfg.stage1_data_dir)
    per_combo = cfg.n_eval_struct // len(combos) if combos else 0
    target = cfg.n_train + cfg.n_eval_indist

    main: dict[str, Query] = {}
    struct: list[dict[str, Query]] = [{} for _ in combos]
    n_id = n_empty = n_stage1 = 0
    s_id = [0] * len(combos)
    s_empty = [0] * len(combos)

    main_open = True

    def try_add(q: Query | None) -> bool:
        nonlocal n_id, n_empty, n_stage1
        if q is None:
            return False
        sql = q.render()
        if sql in main or any(sql in s for s in struct):
            return False
        hit = struct_hit(q, combos)
        if hit is not None:
            if sql in s1_train:
                return False
            # 구조 평가에도 같은 상한을 조합별로 적용
            if (len(struct[hit]) < per_combo
                    and not (q.meta["has_id"] and s_id[hit] + 1 > cfg.id_ratio_max * per_combo)
                    and not (is_empty(q) and s_empty[hit] + 1 > cfg.empty_ratio_max * per_combo)):
                struct[hit][sql] = q
                s_id[hit] += q.meta["has_id"]
                s_empty[hit] += is_empty(q)
            return False
        if not main_open or sql in s1_eval:
            return False
        if q.meta["has_id"] and n_id + 1 > cfg.id_ratio_max * target:
            return False
        if is_empty(q) and n_empty + 1 > cfg.empty_ratio_max * target:
            return False
        if is_stage1_shape(q) and n_stage1 + 1 > cfg.stage1_ratio_max * target:
            return False
        main[sql] = q
        n_id += q.meta["has_id"]
        n_empty += is_empty(q)
        n_stage1 += is_stage1_shape(q)
        return True

    # 1. 커버리지 하한
    for cell in cells():
        if cell_fully_held(cell, combos):
            continue
        got = tries = 0
        while got < cfg.cell_floor and tries < cfg.cell_floor * 100:
            tries += 1
            got += try_add(sample_query(rng, pools, cfg, cell=cell))

    # 1-2. 고정값 표현 하한 (L16)
    for key, floor in cfg.value_floors.items():
        pin = parse_value_key(key)
        got = tries = 0
        while got < floor and tries < floor * 200:
            tries += 1
            got += try_add(sample_query(rng, pools, cfg, pin=pin))

    # 1-3. 존재 확인 하한 (L1)
    got = tries = 0
    while got < cfg.exists_floor and tries < cfg.exists_floor * 200:
        tries += 1
        got += try_add(sample_query(rng, pools, cfg, exists=True))

    # 2. 목표 분포
    tries = 0
    while len(main) < target and tries < target * 100:
        tries += 1
        try_add(sample_query(rng, pools, cfg))
    if len(main) < target:
        raise RuntimeError(f"학습+분포 내 평가 SQL {len(main)}개 < 목표 {target}개 (상한·풀 크기 확인)")

    # 구조 홀드아웃 칸이 덜 찼으면 마저 채운다 (학습 풀은 닫음)
    main_open = False
    tries = 0
    while any(len(s) < per_combo for s in struct) and tries < cfg.n_eval_struct * 200:
        tries += 1
        try_add(sample_query(rng, pools, cfg))

    # 3. 분할
    rows = list(main.values())
    rng.shuffle(rows)
    indist = [q for q in rows if q.render() not in s1_train][:cfg.n_eval_indist]
    indist_sql = {q.render() for q in indist}
    train = [q for q in rows if q.render() not in indist_sql]
    taken = set(main) | {sql for s in struct for sql in s}
    if cfg.pair_floor:
        _plain_pair_floor(train, indist, rng, pools, cfg, combos, s1_eval, taken)
    out = {
        "indist": indist,
        "train": train,
        "struct": [_with_meta(q, struct_combo=i) for i, s in enumerate(struct) for q in s.values()],
        "holdout": generate_holdout(rng, pools, ho, cfg, combos, taken | s1_train),
    }
    return out


def _value_pinned(q: Query, cfg: GenConfig) -> bool:
    keys = [parse_value_key(k) for k in cfg.value_floors]
    return any(q.table == t and c.col == col and c.op == op and c.val == val
               for t, col, op, val in keys for c in q.conds)


def _plain_pair_floor(train: list[Query], indist: list[Query], rng: random.Random, pools: Pools, cfg: GenConfig,
                      combos: list[set[tuple]], s1_eval: set[str], taken: set[str]) -> dict:
    """일반 조회 학습 SQL 의 조합 하한. 칸 하한·존재 확인·고정값 하한과 ID·0행·1단계 모양 상한을 지킨다."""
    target = cfg.n_train + cfg.n_eval_indist
    main = train + indist
    caps = {"id": (lambda q: q.meta["has_id"], cfg.id_ratio_max), "empty": (is_empty, cfg.empty_ratio_max),
            "stage1": (is_stage1_shape, cfg.stage1_ratio_max)}
    n = {k: sum(bool(f(q)) for q in main) for k, (f, _) in caps.items()}
    cell_n = Counter((q.table, c.col, c.op) for q in train for c in q.conds)

    def removable(old, new):
        if is_exists(old) or _value_pinned(old, cfg):
            return False
        lost = Counter((old.table, c.col, c.op) for c in old.conds)
        lost.subtract((new.table, c.col, c.op) for c in new.conds)
        if any(k > 0 and cell_n[cell] - k < cfg.cell_floor_train for cell, k in lost.items()):
            return False
        return all(n[k] - bool(f(old)) + bool(f(new)) <= cap * target for k, (f, cap) in caps.items())

    def on_swap(old, new):
        for k, (f, _) in caps.items():
            n[k] += bool(f(new)) - bool(f(old))
        cell_n.subtract((old.table, c.col, c.op) for c in old.conds)
        cell_n.update((new.table, c.col, c.op) for c in new.conds)
        taken.add(new.render())

    def sample(pair):
        q = sample_query(rng, pools, cfg, cell=_plain_hint(rng, pair))
        if q is None:
            return None
        sql = q.render()
        if sql in taken or sql in s1_eval or struct_hit(q, combos) is not None:
            return None
        return q

    return ensure_pair_floor(train, plain_features, sample, cfg.pair_floor, rng, removable, on_swap,
                             stratum=lambda q: (q.table, len(q.conds)))


def _with_meta(q: Query, **kw) -> Query:
    return replace(q, meta={**q.meta, **kw})


def holdout_literals(ho: dict) -> list[tuple[str, str, str, object]]:
    """holdout.json -> (tier, table, col, value). 1단계 sql_gen.holdout_literals 와 같은 규칙."""
    out: list[tuple[str, str, str, object]] = []
    for r in ho["customers"]["heldout_rows"]:
        out += [("unseen_entity", "customers", "name", r["name"]),
                ("unseen_entity", "customers", "customer_id", r["customer_id"]),
                ("unseen_entity", "orders", "customer_id", r["customer_id"])]
    for r in ho["items"]["heldout_rows"]:
        out += [("unseen_entity", "items", "item_name", r["item_name"]),
                ("unseen_entity", "items", "item_id", r["item_id"]),
                ("unseen_entity", "orders", "item_id", r["item_id"])]
    for cid in ho["customers"]["heldout_ids_only"]:
        out += [("unseen_id", "customers", "customer_id", cid), ("unseen_id", "orders", "customer_id", cid)]
    for iid in ho["items"]["heldout_ids_only"]:
        out += [("unseen_id", "items", "item_id", iid), ("unseen_id", "orders", "item_id", iid)]
    for oid in ho["orders"]["heldout_ids"]:
        out.append(("unseen_order_id", "orders", "order_id", oid))
    for value in ho["orders"].get("heldout_dates", []):
        out.append(("unseen_date", "orders", "order_date", value))
    return out


HOLDOUT_SHAPES = {
    **{f"date_{op}": {"n": 1, "conn": None, "order": False, "limit": False, "date_op": op}
       for op in ("<", ">", "<=", ">=", "BETWEEN")},
    "single":      {"n": 1, "conn": None,  "order": False, "limit": False},
    "in2":         {"n": 1, "conn": None,  "order": False, "limit": False, "in": 2},
    "in3":         {"n": 1, "conn": None,  "order": False, "limit": False, "in": 3},
    "in2_and":     {"n": 2, "conn": "AND", "order": False, "limit": False, "in": 2},
    "and2":        {"n": 2, "conn": "AND", "order": False, "limit": False},
    "and3":        {"n": 3, "conn": "AND", "order": False, "limit": False},
    "order":       {"n": 1, "conn": None,  "order": True,  "limit": False},
    "order_limit": {"n": 1, "conn": None,  "order": True,  "limit": True},
    "and2_order":  {"n": 2, "conn": "AND", "order": True,  "limit": False},
}


def generate_holdout(rng: random.Random, pools: Pools, ho: dict, cfg: GenConfig,
                     combos: list[set[tuple]], seen: set[str]) -> list[Query]:
    """미등장 리터럴마다 들어갈 수 있는 모양을 전부 하나씩 만든다 (비율이 아니라 능력 시험이 목적).

    이 값 holdout 버전에서는 유일 키를 단독(= / IN)으로 샘플링하고,
    orders 의 FK 는 AND·IN·정렬·LIMIT 까지 시험한다. 유효 SQL 전체의 제한은 아니다.
    결과 0행, 구조 홀드아웃 조합은 피한다 (값 복사와 구조 일반화를 섞지 않으려고).
    """
    rows: list[Query] = []
    seen = set(seen)
    for tier, table, col, value in holdout_literals(ho):
        unique = (table, col) in schema.UNIQUE_KEYS
        shapes = (["single", "date_<", "date_>", "date_<=", "date_>=", "date_BETWEEN", "and2", "order_limit"]
                  if col == "order_date" else cfg.holdout_shapes_unique if unique else cfg.holdout_shapes_fk)
        for name in shapes:
            for _ in range(300):
                q = sample_query(rng, pools, cfg, fixed=(table, col, value), shape=HOLDOUT_SHAPES[name])
                if (q is None or struct_hit(q, combos) is not None
                        or q.meta["n_rows"] == 0 or q.render() in seen):
                    continue
                seen.add(q.render())
                rows.append(_with_meta(q, holdout_tier=tier, holdout_shape=name))
                break
    return rows


# ---------------------------------------------------------------------------
# 저장 · 리포트
# ---------------------------------------------------------------------------

DEFAULT_OUT = "data/stage3/sql"
FILES = {"train": "train.json", "indist": "eval_indist.json",
         "struct": "eval_struct.json", "holdout": "eval_holdout.json"}
REPORT_FILE = "gen_report.json"


def stats(qs: list[Query]) -> dict:
    n = len(qs) or 1
    cnt = lambda key: dict(sorted(Counter(key(q) for q in qs).items(), key=lambda kv: str(kv[0])))
    return {
        "n": len(qs),
        "n_skeletons": len({skeleton(q) for q in qs}),
        "n_conds": cnt(lambda q: len(q.conds)),
        "conn": cnt(lambda q: q.conn or "-"),
        "order": cnt(lambda q: q.order is not None),
        "limit": cnt(lambda q: "-" if q.limit is None else ("1" if q.limit == 1 else "n")),
        "table": cnt(lambda q: q.table),
        "select_kind": cnt(lambda q: schema.select_kind(q.table, q.select, q.cond_cols())),
        "ops": dict(sorted(Counter(op for q in qs for op in q.ops()).items())),
        "id_ratio": round(sum(q.meta["has_id"] for q in qs) / n, 4),
        "empty_ratio": round(sum(is_empty(q) for q in qs) / n, 4),
        "stage1_ratio": round(sum(is_stage1_shape(q) for q in qs) / n, 4),
        "exists_ratio": round(sum(is_exists(q) for q in qs) / n, 4),
        "date_where_ratio": round(sum(any(c.col == "order_date" for c in q.conds) for q in qs) / n, 4),
        "date_select_ratio": round(sum(q.select == "order_date" for q in qs) / n, 4),
        "date_order_ratio": round(sum(bool(q.order) and q.order[0] == "order_date" for q in qs) / n, 4),
        "date_any_ratio": round(sum(q.select == "order_date" or any(c.col == "order_date" for c in q.conds)
                                    or bool(q.order) and q.order[0] == "order_date" for q in qs) / n, 4),
    }


def cell_counts(qs: list[Query]) -> dict[str, int]:
    c = Counter(f"{q.table}.{x.col} {x.op}" for q in qs for x in q.conds)
    return {f"{t}.{col} {op}": c.get(f"{t}.{col} {op}", 0) for t, col, op in cells()}


def build(db_dir: str, out_dir: str, cfg: GenConfig) -> dict:
    con, ho, _ = _open(db_dir)
    parts = generate(con, ho, cfg)
    con.close()
    os.makedirs(out_dir, exist_ok=True)
    for name, qs in parts.items():
        with open(os.path.join(out_dir, FILES[name]), "w", encoding="utf-8") as f:
            json.dump([to_dict(q) for q in qs], f, ensure_ascii=False, indent=1)
    report = {
        "config": asdict(cfg), "db_dir": db_dir,
        "stats": {name: stats(qs) for name, qs in parts.items()},
        "train_cells": cell_counts(parts["train"]),
        "struct_per_combo": dict(Counter(q.meta["struct_combo"] for q in parts["struct"])),
        "holdout_per_tier": dict(Counter(q.meta["holdout_tier"] for q in parts["holdout"])),
        "value_counts": {key: sum(1 for q in parts["train"] if q.table == t and any(
            c.col == col and c.op == op and c.val == val for c in q.conds))
            for key, (t, col, op, val) in ((k, parse_value_key(k)) for k in cfg.value_floors)},
        "holdout_tier_shape": {f"{t}|{sh}": n for (t, sh), n in sorted(Counter(
            (q.meta["holdout_tier"], q.meta["holdout_shape"]) for q in parts["holdout"]).items())},
    }
    if cfg.n_aggregate_train:
        report["aggregates"] = _build_aggregates(db_dir, Path(out_dir), cfg.seed,
                                                  cfg.n_aggregate_train, cfg.n_aggregate_eval, cfg.pair_floor)
    if cfg.pair_floor:
        report["pair_floor"] = {"floor": cfg.pair_floor,
                                "short": short_pairs(parts["train"], plain_features, cfg.pair_floor)}
    with open(os.path.join(out_dir, REPORT_FILE), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def _load(out_dir: str) -> tuple[dict, dict[str, list[dict]]]:
    report = json.load(open(os.path.join(out_dir, REPORT_FILE), encoding="utf-8"))
    parts = {name: json.load(open(os.path.join(out_dir, fn), encoding="utf-8")) for name, fn in FILES.items()}
    parts = {name: [r for r in rows if not r.get("query_kind")] for name, rows in parts.items()}
    return report, parts


def _query(d: dict) -> Query:
    meta = {k: d[k] for k in ("n_rows", "is_nonexistent", "has_id") if k in d}
    q = from_dict(d)
    return replace(q, meta=meta)


def verify(out_dir: str, check_determinism: bool = True) -> list[str]:
    report, parts = _load(out_dir)
    cfg = GenConfig(**report["config"])
    con, ho, bans = _open(report["db_dir"])
    combos = combos_of(cfg)
    fails: list[str] = []

    # 1. 실행 · n_rows · 정규형 왕복 · 구조 규칙
    for name, rows in parts.items():
        for d in rows:
            try:
                n = len(con.execute(d["sql"]).fetchall())
            except sqlite3.Error as e:
                fails.append(f"[{name}] 실행 오류 {d['sql']!r}: {e}")
                continue
            if n != d["n_rows"]:
                fails.append(f"[{name}] n_rows 불일치 {d['sql']!r}: {d['n_rows']} != {n}")
            q = from_dict(d)
            if q.render() != d["sql"] or parse(d["sql"]).render() != d["sql"]:
                fails.append(f"[{name}] 정규형 왕복 실패 {d['sql']!r}")
            if v := violations(q):
                fails.append(f"[{name}] 규칙 위반 {d['sql']!r}: {v}")

    # 2. 미등장 리터럴: 학습·분포 내·구조 평가에는 없고, holdout 에는 반드시 있음
    def banned_conds(d: dict) -> list:
        out = []
        for c in d["conds"]:
            b = bans.get(d["table"], {}).get(c["col"], set())
            if c["op"] not in ("IN", "LIKE") and (c["val"] in b or (c["op"] == "BETWEEN" and c["val2"] in b)):
                out.append(c)
            if c["op"] == "IN" and set(c["val"]) & b:
                out.append(c)
            if c["op"] == "LIKE" and c["val"].strip("%") in b:
                out.append(c)
        return out
    for name in ("train", "indist", "struct"):
        for d in parts[name]:
            if banned_conds(d):
                fails.append(f"[{name}] 미등장 리터럴 사용 {d['sql']!r}")
    for d in parts["holdout"]:
        if not banned_conds(d):
            fails.append(f"[holdout] 미등장 리터럴 없음 {d['sql']!r}")
        if d["n_rows"] == 0:
            fails.append(f"[holdout] 결과 0행 {d['sql']!r}")

    # 3. 분할 간 SQL 비중복 + 분할 내 중복 없음
    sets = {name: [d["sql"] for d in rows] for name, rows in parts.items()}
    for name, sqls in sets.items():
        if len(set(sqls)) != len(sqls):
            fails.append(f"[{name}] 중복 SQL 존재")
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ov := set(sets[a]) & set(sets[b]):
                fails.append(f"{a} / {b} SQL 중복 {len(ov)}건 (예: {next(iter(ov))!r})")

    # 4. 1단계와 겹침: 1단계 평가 SQL 은 학습에 없고, 1단계 학습 SQL 은 평가 분할에 없음
    s1_train, s1_eval = stage1_sqls(cfg.stage1_data_dir)
    if ov := set(sets["train"]) & s1_eval:
        fails.append(f"[train] 1단계 평가 SQL {len(ov)}건 (예: {next(iter(ov))!r})")
    for name in ("indist", "struct", "holdout"):
        if ov := set(sets[name]) & s1_train:
            fails.append(f"[{name}] 1단계 학습 SQL {len(ov)}건 (예: {next(iter(ov))!r})")

    # 5. 구조 홀드아웃: 학습·분포 내에 그 조합이 없고, 조합의 각 요소는 학습에 있음
    train_q = [_query(d) for d in parts["train"]]
    main_q = train_q + [_query(d) for d in parts["indist"]]
    for q in main_q:
        if struct_hit(q, combos) is not None:
            fails.append(f"구조 홀드아웃 조합이 학습/분포 내에 있음 {q.render()!r}")
    train_feats = set().union(*(features(q) for q in train_q)) if train_q else set()
    for combo in combos:
        for atom in combo:
            if atom not in train_feats:
                fails.append(f"구조 홀드아웃 요소 {atom} 가 학습에 한 번도 없음")
    for i in range(len(combos)):
        if not any(d["struct_combo"] == i for d in parts["struct"]):
            fails.append(f"구조 홀드아웃 조합 {cfg.struct_holdout[i]} 의 평가 SQL 이 없음")

    # 6. 상한 (학습 + 분포 내)
    st = stats(main_q)
    for key, cap in (("id_ratio", cfg.id_ratio_max), ("empty_ratio", cfg.empty_ratio_max),
                     ("stage1_ratio", cfg.stage1_ratio_max)):
        if st[key] > cap + 1e-9:
            fails.append(f"{key} {st[key]:.1%} > 상한 {cap:.0%}")
    st_struct = stats([_query(d) for d in parts["struct"]])
    for key, cap in (("id_ratio", cfg.id_ratio_max), ("empty_ratio", cfg.empty_ratio_max)):
        if st_struct[key] > cap + 1e-9:
            fails.append(f"[struct] {key} {st_struct[key]:.1%} > 상한 {cap:.0%}")

    # 7. 커버리지 하한 (학습)
    cc = cell_counts(train_q)
    for cell in cells():
        key = f"{cell[0]}.{cell[1]} {cell[2]}"
        if not cell_fully_held(cell, combos) and cc[key] < cfg.cell_floor_train:
            fails.append(f"커버리지 부족 {key}: 학습 {cc[key]}개 < {cfg.cell_floor_train}")

    # 7-2. 고정값 표현 하한 (학습)
    for key in cfg.value_floors:
        t, col, op, val = parse_value_key(key)
        k = sum(1 for q in train_q if q.table == t and any(c.col == col and c.op == op and c.val == val for c in q.conds))
        if k < cfg.value_floor_train:
            fails.append(f"고정값 표현 부족 {key}: 학습 {k}개 < {cfg.value_floor_train}")

    # 7-3. 존재 확인 하한 (학습)
    if (k := sum(map(is_exists, train_q))) < cfg.exists_floor_train:
        fails.append(f"존재 확인(L1) 부족: 학습 {k}개 < {cfg.exists_floor_train}")

    # 7-4. 조합 하한 (학습): 하한보다 적은 쌍은 생성 때 채우지 못해 리포트에 남긴 것뿐이어야 한다
    if cfg.pair_floor:
        short = short_pairs(train_q, plain_features, cfg.pair_floor)
        if short != report.get("pair_floor", {}).get("short"):
            fails.append(f"조합 하한 리포트와 학습 SQL 이 다름 (하한 미달 {len(short)}쌍)")

    # 8. 결정성
    if check_determinism:
        regen = generate(con, ho, cfg)
        for name, qs in regen.items():
            if [q.render() for q in qs] != sets[name]:
                fails.append(f"[{name}] 같은 설정으로 다시 만든 결과가 저장본과 다름")
    con.close()
    if report.get("aggregates"):
        try:
            _verify_aggregates(out_dir)
        except (AssertionError, ValueError, KeyError, OSError, sqlite3.Error) as exc:
            fails.append(f"[집계] 검증 실패: {exc}")
    elif report["config"].get("n_aggregate_train", 0):
        fails.append("집계 SQL 분할이 없음")
    return fails


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------

def summary(out_dir: str) -> None:
    report = json.load(open(os.path.join(out_dir, REPORT_FILE), encoding="utf-8"))
    for name, st in report["stats"].items():
        print(f"[{name}] {st['n']}개, 골격 {st['n_skeletons']}종 | ID {st['id_ratio']:.1%} · "
              f"0행 {st['empty_ratio']:.1%} · 1단계 모양 {st['stage1_ratio']:.1%}")
        for key in ("n_conds", "conn", "order", "limit", "table", "select_kind", "ops",
                    "date_where_ratio", "date_select_ratio", "date_order_ratio", "date_any_ratio"):
            print(f"    {key:<12} {st[key]}")
    print("\n[train] (테이블.컬럼 연산자) 칸별 개수")
    for k, v in report["train_cells"].items():
        print(f"    {k:<28}{v:>7}")
    combos = report["config"]["struct_holdout"]
    print("\n[struct] 조합별 개수")
    for i, combo in enumerate(combos):
        print(f"    {combo}: {report['struct_per_combo'].get(str(i), report['struct_per_combo'].get(i, 0))}")
    print(f"\n[holdout] 층별 개수 {report['holdout_per_tier']}")
    if report.get("value_counts"):
        print(f"[train] 고정값 조건 개수 {report['value_counts']}")
    if report.get("aggregates"):
        counts = report["aggregates"]["counts"]
        print(f"[집계·그룹·서브쿼리] {counts}")


# ---------------------------------------------------------------------------

def _aggregate_family(query):
    if query.subquery:
        return "scalar_subquery"
    return "having" if query.having else "group" if query.group_by else "aggregate"


def _aggregate_struct_combo(query):
    if query.group_by and query.having and query.limit:
        return "having_limit"
    if query.group_by == "order_date" and query.aggregate.startswith("AVG"):
        return "date_group_avg"
    if query.subquery and query.inner_conds and any(c.col == "order_date" for c in query.conds):
        return "date_outer_filtered_subquery"
    return None


def aggregate_features(query):
    """집계 계열의 조합 하한 특징: 계열·테이블·함수·그룹 키·HAVING·정렬·LIMIT, 바깥/안쪽 조건, 서브쿼리 비교·출력."""
    f = {("fam", _aggregate_family(query)), ("tab", query.table), ("nc", len(query.conds)),
         ("ord", query.order[1] if query.order else "-"), ("lim", query.limit is not None)}
    f |= _cond_features(query.table, query.conds)
    if query.aggregate:
        f |= {("fn", query.aggregate.split("(")[0]), ("agg", query.aggregate)}
    if query.group_by:
        f.add(("grp", query.group_by))
    if query.having:
        f.add(("hop", query.having[0]))
    if query.subquery:
        col, op, inner = query.subquery
        projection = ("*" if query.projection == "*" else "ident" if query.projection == schema.IDENT[query.table]
                      else "col")
        f |= {("sop", op), ("sfn", inner.split("(")[0]), ("scol", col), ("proj", projection),
              ("inc", len(query.inner_conds))} | _cond_features(query.table, query.inner_conds, "icond")
    return f


SUBQUERY_FEATURES = {"sop", "sfn", "scol", "proj", "inc", "icond"}


def _aggregate_hint(rng, pair):
    """쌍을 겨냥한 샘플링 인자 (계열, 테이블, 바깥·안쪽 조건 수)."""
    feats = {}
    for k, v in pair:
        feats.setdefault(k, []).append(v)
    kind = feats["fam"][0] if "fam" in feats else None
    if kind is None and SUBQUERY_FEATURES & set(feats):
        kind = "scalar_subquery"
    elif kind is None and "hop" in feats:
        kind = "having"
    elif kind is None and ({"grp"} & set(feats) or feats.get("ord", ["-"])[0] != "-" or feats.get("lim") == [True]):
        kind = "group"
    columns = [str(v).split("(")[-1].rstrip(")") for k in ("col", "grp", "scol", "agg") for v in feats.get(k, [])]
    tables = [t for t in sql_ast.GROUP_COLUMNS if all(c in schema.TABLE_COLUMNS[t] for c in columns if c != "*")
              and all(g in sql_ast.GROUP_COLUMNS[t] for g in feats.get("grp", []))]
    if "tab" in feats:
        tables = [feats["tab"][0]]
    if kind == "scalar_subquery":
        tables = [t for t in tables if sql_ast.NUMERIC_COLUMNS[t]]
    n_outer = feats["nc"][0] if "nc" in feats else (min(2, len(feats.get("cond", [])) + len(feats.get("col", [])))
                                                    or None)
    n_inner = feats["inc"][0] if "inc" in feats else (len(feats["icond"]) if "icond" in feats else None)
    if n_outer == 1 and rng.random() < .5 and "nc" not in feats:
        n_outer = 2
    return {"kind": kind, "table": rng.choice(tables) if tables else None, "n_conds": n_outer, "n_inner": n_inner,
            "outer_spec": tuple(feats.get("cond", ())), "inner_spec": tuple(feats.get("icond", ()))}


def _aggregate_filters(rng, pools, table, excluded=(), n=None, spec=()):
    # Include ordinary row filters; pools already exclude heldout literals.
    # spec: 반드시 넣을 조건 종류 ("date:BETWEEN", "num:cmp" …, 조합 하한용). 나머지는 무작위
    eligible = [c for c in schema.TABLE_COLUMNS[table]
                if c not in excluded]
    forced = []
    for kind_op in spec:
        options = [(c, op) for c in eligible if c not in {f[0] for f in forced} for op in schema.ops(table, c)
                   if f"{schema.kind(table, c)}:{_op_class(op)}" == kind_op]
        if not options:
            return None
        forced.append(rng.choice(options))
    rest = [c for c in eligible if c not in {f[0] for f in forced}]
    k = rng.choices([0, 1, 2], [30, 50, 20])[0] if n is None else n
    chosen = [(c, None) for c in rng.sample(rest, max(0, min(k - len(forced), len(rest))))]
    result = []
    for col, op in forced + chosen:
        op = op or _pick_op(rng, table, col, "AND", GenConfig())
        cond, _ = _make_cond(rng, pools, table, col, op, GenConfig())
        if cond is not None:
            result.append(cond)
    return tuple(sorted(result, key=lambda c: schema.col_index(table, c.col)))


def _sample_aggregate_query(rng, pools, kind=None, forced=None, table=None, n_conds=None, n_inner=None,
                            outer_spec=(), inner_spec=()):
    table = forced[0] if forced else table or rng.choice(list(sql_ast.GROUP_COLUMNS))
    numbers = sql_ast.NUMERIC_COLUMNS[table]
    kind = kind or rng.choices(["aggregate", "group", "having", "scalar_subquery"], [30, 25, 25, 20])[0]
    if kind == "scalar_subquery" and not numbers:
        return None
    conds = _aggregate_filters(rng, pools, table, n=n_conds, spec=outer_spec)
    if conds is None:
        return None
    if forced:
        conds = (Cond(forced[1], "=", forced[2]),)
    if kind == "scalar_subquery":
        col = rng.choice(numbers)
        fn = rng.choice(["AVG", "MIN", "MAX"])
        query = sql_ast.AdvancedQuery(table, conds=conds, projection=rng.choice(["*", schema.IDENT[table], col]),
                                subquery=(col, rng.choice(["<", ">", "<=", ">=", "="]), f"{fn}({col})"),
                                inner_conds=_aggregate_filters(rng, pools, table, n=n_inner, spec=inner_spec) or ())
        if inner_spec and len(query.inner_conds) < len(inner_spec):
            return None
    else:
        aggregates = ["COUNT(*)"] + [f"{fn}({col})" for col in numbers for fn in ("SUM", "AVG", "MIN", "MAX")]
        if table == "orders" and kind != "having":
            aggregates += ["MIN(order_date)", "MAX(order_date)"]
        aggregate = rng.choice(aggregates)
        group = rng.choice(sql_ast.GROUP_COLUMNS[table]) if kind != "aggregate" else None
        query = sql_ast.AdvancedQuery(table, aggregate, conds, group)
        if sql_ast.violations(query):
            return None
        raw_rows = pools.con.execute(query.render()).fetchall()
        if kind == "having":
            values = sorted({float(row[-1]) for row in raw_rows if row[-1] is not None})
            threshold = rng.choice(values or [0.0])
            query = replace(query, having=(rng.choice(["=", "!=", ">", ">=", "<", "<="]), round(threshold, 2)))
        if group and rng.random() < .5:
            query = replace(query, order=(aggregate, rng.choice(["ASC", "DESC"])))
            if rng.random() < .6:
                query = replace(query, limit=rng.choice([1, 3, 5, 10]))
    query = sql_ast.canonical(query)
    if sql_ast.violations(query):
        return None
    rows = pools.con.execute(query.render()).fetchall()
    result_kind = ("empty" if not rows else "null" if any(row[-1] is None for row in rows)
                   else "zero_count" if query.aggregate == "COUNT(*)" and not query.group_by and rows == [(0,)]
                   else "nonempty")
    return replace(query, meta={"n_rows": len(rows), "result_kind": result_kind,
                                "family": _aggregate_family(query), "struct_combo": _aggregate_struct_combo(query),
                                "has_id": any(c.col.endswith("_id") for c in (*query.conds, *query.inner_conds)),
                                "is_nonexistent": any(schema.kind(table, c.col) == "id" and pools.cond_count(table, c) == 0
                                                      for c in (*query.conds, *query.inner_conds))})


def _aggregate_banned_values(query, bans):
    return [(c.col, v) for c in (*query.conds, *query.inner_conds)
            for v in (c.val if c.op == "IN" else [c.val, c.val2] if c.op == "BETWEEN" else [c.val])
            if v in bans[query.table].get(c.col, set())]


def _aggregate_coverage_requirements():
    """Atomic training coverage needed to interpret the structural holdouts."""
    requirements = {f"family:{name}": lambda q, name=name: _aggregate_family(q) == name
                    for name in ("aggregate", "group", "having", "scalar_subquery")}
    requirements.update({f"function:{fn}": lambda q, fn=fn: bool(q.aggregate and q.aggregate.startswith(fn + "("))
                         for fn in ("COUNT", "SUM", "AVG", "MIN", "MAX")})
    requirements.update({
        "having_without_limit": lambda q: bool(q.having and not q.limit),
        "limit_without_having": lambda q: bool(q.limit and not q.having),
        "date_group": lambda q: q.group_by == "order_date",
        "filtered_subquery": lambda q: bool(q.subquery and q.inner_conds),
        "date_outer_subquery": lambda q: bool(q.subquery and any(c.col == "order_date" for c in q.conds)),
    })
    return requirements


def _ensure_aggregate_training_coverage(training, rng, pools, seen):
    """Coverage is a sampling requirement, not a prohibition of valid SQL."""
    requirements = _aggregate_coverage_requirements()
    for name, predicate in requirements.items():
        if any(predicate(q) for q in training):
            continue
        for _ in range(10000):
            query = _sample_aggregate_query(rng, pools)
            if query is None or _aggregate_struct_combo(query) or query.render() in seen or not predicate(query):
                continue
            # Replace only a redundant example; retain all already covered atoms.
            counts = {key: sum(test(q) for q in training) for key, test in requirements.items()}
            index = next((i for i, q in enumerate(training)
                          if all(not test(q) or counts[key] > 1 for key, test in requirements.items())), None)
            if index is None:
                raise ValueError("Training quota too small for structural-holdout coverage")
            training[index] = query
            seen.add(query.render())
            break
        else:
            raise RuntimeError(f"Could not sample training coverage: {name}")


def _aggregate_pair_floor(training, rng, pools, seen, floor):
    """집계 계열 학습 SQL 의 조합 하한. 구조 홀드아웃 원자 커버리지(_aggregate_coverage_requirements)를 지킨다."""
    requirements = _aggregate_coverage_requirements()
    covered = Counter(key for q in training for key, test in requirements.items() if test(q))

    def removable(old, new):
        return all(covered[key] > 1 or test(new) for key, test in requirements.items() if test(old))

    def on_swap(old, new):
        covered.subtract(key for key, test in requirements.items() if test(old))
        covered.update(key for key, test in requirements.items() if test(new))
        seen.add(new.render())

    def sample(pair):
        query = _sample_aggregate_query(rng, pools, **_aggregate_hint(rng, pair))
        if query is None or _aggregate_struct_combo(query) or query.render() in seen:
            return None
        return query

    return ensure_pair_floor(training, aggregate_features, sample, floor, rng, removable, on_swap,
                             stratum=_aggregate_family)


def _build_aggregates(db_dir, out, seed=0, n_train=20000, n_eval=1000, pair_floor=0):
    out = Path(out)
    con, ho, bans = _open(str(db_dir))
    cfg = db_gen.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in ho["config"].items()})
    pools = Pools(con, bans, cfg)
    rng = random.Random(seed)
    main, structural, seen = [], [], set()
    counts = Counter()
    attempts = 0
    while len(main) < n_train + n_eval or len(structural) < n_eval:
        attempts += 1
        if attempts > (n_train + n_eval) * 150:
            raise RuntimeError("Could not fill advanced SQL quotas")
        query = _sample_aggregate_query(rng, pools)
        if query is None or query.render() in seen:
            continue
        combo = _aggregate_struct_combo(query)
        if combo:
            # Keep structural combinations balanced and all atoms in training.
            index = ["having_limit", "date_group_avg", "date_outer_filtered_subquery"].index(combo)
            cap = n_eval // 3 + (index < n_eval % 3)
            if len(structural) < n_eval and counts[combo] < cap:
                structural.append(query)
                counts[combo] += 1
                seen.add(query.render())
        elif len(main) < n_train + n_eval:
            main.append(query)
            seen.add(query.render())
    rng.shuffle(main)
    training = main[n_eval:]
    _ensure_aggregate_training_coverage(training, rng, pools, seen)
    floor_report = _aggregate_pair_floor(training, rng, pools, seen, pair_floor) if pair_floor else None
    held = []
    for tier, table, col, value in holdout_literals(ho):
        for _ in range(80):
            query = _sample_aggregate_query(rng, pools, forced=(table, col, value))
            if query and not _aggregate_struct_combo(query) and query.render() not in seen:
                query = replace(query, meta={**query.meta, "holdout_tier": tier})
                held.append(query)
                seen.add(query.render())
                break
    parts = {"train": training, "indist": main[:n_eval], "struct": structural, "holdout": held}
    out.mkdir(parents=True, exist_ok=True)
    for name, queries in parts.items():
        path = out / FILES[name]
        flat = [r for r in json.loads(path.read_text(encoding="utf-8")) if not r.get("query_kind")] if path.exists() else []
        path.write_text(json.dumps(flat + [sql_ast.to_dict(q) for q in queries], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    report = {"seed": seed, "n_train": n_train, "n_eval": n_eval, "db_dir": str(db_dir),
              "counts": {name: len(qs) for name, qs in parts.items()},
              "families": {name: dict(Counter(_aggregate_family(q) for q in qs)) for name, qs in parts.items()},
              "result_profiles": {name: dict(Counter(q.meta["result_kind"] for q in qs)) for name, qs in parts.items()},
              "struct_combos": dict(counts)}
    report["training_coverage"] = {name: sum(predicate(q) for q in training)
                                   for name, predicate in _aggregate_coverage_requirements().items()}
    if floor_report:
        report["pair_floor"] = floor_report
    report_path = out / REPORT_FILE
    combined_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    combined_report["aggregates"] = report
    report_path.write_text(json.dumps(combined_report, indent=2) + "\n", encoding="utf-8")
    con.close()
    return report


def _verify_aggregates(out):
    out = Path(out)
    report = json.loads((out / REPORT_FILE).read_text(encoding="utf-8"))["aggregates"]
    con, ho, bans = _open(report["db_dir"])
    seen, families, training = set(), set(), []
    for split in ("train", "indist", "struct", "holdout"):
        rows = json.loads((out / FILES[split]).read_text(encoding="utf-8"))
        for record in (r for r in rows if r.get("query_kind")):
            query = sql_ast.from_dict(record)
            assert not sql_ast.violations(query), record["sql"]
            assert sql_ast.parse(record["sql"]).render() == query.render() == record["sql"]
            assert query.render() not in seen, "Duplicate SQL across splits"
            seen.add(query.render())
            assert len(con.execute(query.render()).fetchall()) == record["n_rows"]
            assert bool(_aggregate_banned_values(query, bans)) == (split == "holdout"), "Holdout literal leak"
            assert bool(_aggregate_struct_combo(query)) == (split == "struct"), "Structural leak"
            if split == "train":
                families.add(_aggregate_family(query))
                training.append(query)
    assert families == {"aggregate", "group", "having", "scalar_subquery"}
    assert {q.aggregate.split("(")[0] for q in training if q.aggregate} == {"COUNT", "SUM", "AVG", "MIN", "MAX"}
    assert any(q.having and not q.limit for q in training)
    assert any(q.limit and not q.having for q in training)
    assert any(q.group_by == "order_date" for q in training)
    assert any(q.aggregate and q.aggregate.startswith("AVG(") for q in training)
    assert any(q.subquery and q.inner_conds for q in training)
    assert any(q.subquery and any(c.col == "order_date" for c in q.conds) for q in training)
    if "pair_floor" in report:
        floor = report["pair_floor"]
        assert short_pairs(training, aggregate_features, floor["floor"]) == floor["short"], "Pair floor report mismatch"
    con.close()
    return report



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="3단계 SQL 생성기")
    ap.add_argument("command", choices=["build", "verify", "summary"])
    p1 = stage_paths(1)
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"SQL 출력 폴더 (기본 {DEFAULT_OUT})")
    ap.add_argument("--db-dir", default="data/stage3/db",
                    help=f"shop.db·holdout.json 폴더 (기본 data/stage3/db, build 만. verify 는 리포트에 적힌 값)")
    ap.add_argument("--stage1-dir", default=str(p1.data_dir),
                    help=f"겹침 금지용 1단계 쌍 폴더 (train/*.json, eval/<이름>/pairs.json, 기본 {p1.data_dir}, build 만)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=40000, help="일반 조회·날짜 학습 SQL 수")
    ap.add_argument("--n-eval-indist", type=int, default=2000, help="분포 내 평가 SQL 수")
    ap.add_argument("--n-eval-struct", type=int, default=2000, help="구조 일반화 평가 SQL 수 (조합 수로 나눔)")
    ap.add_argument("--cell-floor", type=int, default=200, help="(테이블, 컬럼, 연산자) 칸마다 생성 시 강제로 채우는 수")
    ap.add_argument("--cell-floor-train", type=int, default=150, help="verify: 분할 후 학습에 남아야 하는 칸별 수")
    ap.add_argument("--no-determinism", action="store_true", help="verify: 재생성 비교 생략")
    ap.add_argument("--n-aggregate-train", type=int, default=60000, help="집계·그룹·서브쿼리 학습 SQL 수")
    ap.add_argument("--n-aggregate-eval", type=int, default=1000, help="집계 계열 분포 내·구조 평가 SQL 수")
    ap.add_argument("--pair-floor", type=int, default=50, help="학습 SQL 의 구조 특징 쌍마다 최소 개수 (0 이면 끄기)")
    args = ap.parse_args(argv)
    if args.n_aggregate_train < 1 or args.n_aggregate_eval < 1:
        ap.error("--n-aggregate-train and --n-aggregate-eval must be positive")

    if args.command == "build":
        cfg = GenConfig(seed=args.seed, n_train=args.n_train, n_eval_indist=args.n_eval_indist,
                        n_eval_struct=args.n_eval_struct, cell_floor=args.cell_floor,
                        cell_floor_train=args.cell_floor_train, stage1_data_dir=args.stage1_dir,
                        n_aggregate_train=args.n_aggregate_train, n_aggregate_eval=args.n_aggregate_eval,
                        pair_floor=args.pair_floor)
        report = build(args.db_dir, args.out, cfg)
        for name, st in report["stats"].items():
            print(f"생성  {os.path.join(args.out, FILES[name])} ({st['n']}개)")
        print(f"집계·그룹·서브쿼리 생성  {report['aggregates']['counts']}")
        fails = verify(args.out, check_determinism=not args.no_determinism)
        print("검증 ", "통과" if not fails else "실패\n  - " + "\n  - ".join(fails[:30]))
        return 1 if fails else 0

    if args.command == "verify":
        fails = verify(args.out, check_determinism=not args.no_determinism)
        print("통과" if not fails else "실패\n  - " + "\n  - ".join(fails[:30]))
        return 1 if fails else 0

    summary(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
