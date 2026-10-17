"""
Text-to-SQL 2단계 SQL 생성기

1단계 shop.db / holdout.json 을 그대로 쓰고, 2단계 문법(sql_ast.py)의 SQL을 생성한다.

절차
  1. 커버리지 하한: (테이블, 컬럼, 연산자) 칸마다 cell_floor 개를 그 칸을 포함하도록 강제 샘플링
     고정값 표현 하한: value_floors 의 조건(stock = 0 / stock > 0, 라벨 규칙 L16)을 포함한 SQL 을 정한 개수만큼
     존재 확인 하한: L1(SELECT = 조건 컬럼, 조건 하나 =) SQL 을 exists_floor 개 (1단계에서 존재 확인 질문이 적어 틀렸다)
  2. 목표 분포: 조건 수 / IN / AND·OR / ORDER BY / LIMIT 비중에 따라 나머지를 샘플링
     - IN 은 같은 컬럼의 값 2~3개 나열(조건 하나), OR 는 서로 다른 컬럼 조건끼리(ID·이름 조건 제외)
     - 상한(caps): ID 조건 포함 SQL, 결과 0행 SQL(없는 ID 제외), 1단계 모양 SQL(조건 1개 = , 정렬 없음, 존재 확인 제외)
     - 조건 하나만으로 0행이거나 전체 행이 나오는(무의미한) 조건은 버림
  3. 분할
     - struct : struct_holdout 조합에 걸리는 SQL. 학습·분포 내 평가에 절대 넣지 않음 (구조 일반화 평가)
     - train / indist : 나머지를 SQL 단위로 분할
     - holdout : holdout.json 리터럴을 = 또는 IN 조건 하나에 넣은 SQL (미등장 값 평가)
  4. 1단계와 겹침 금지 (1단계 쌍을 2단계 학습에 섞고, 1단계 평가셋 점수도 계속 재기 때문)
     - 1단계 평가 SQL 은 학습·분포 내 평가에 넣지 않음
     - 1단계 학습 SQL 은 2단계 평가 분할(indist / struct / holdout)에 넣지 않음

사용법 (저장소 루트에서)
    python -m src.data.stage2.sql_gen build     # data/stage2/sql/ 에 SQL 목록 + 리포트
    python -m src.data.stage2.sql_gen build --n-train 2000 --n-eval-indist 200 --n-eval-struct 200 --out data_raw/stage2/sql_pilot
    python -m src.data.stage2.sql_gen verify    # 완료 조건 재검사 (결정성 포함)
    python -m src.data.stage2.sql_gen summary   # 분포 표
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import stage_paths
from src.data import dataset, db_gen
from src.evaluation import eval_sets
from src.data.stage2 import schema
from src.data.stage2.sql_ast import Cond, Query, canonical, features, from_dict, parse, skeleton, to_dict, violations

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
    # 유일 키(이름·PK)의 IN 비율. 유일 키 = 는 단독(1단계 모양, 상한 5%)으로만 나와서, 이름 여러 개를 묻는 SQL 은 IN 이 맡는다
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
    exists_floor_train: int = 850  # verify: 분할 후 학습에 남아야 하는 개수

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
            self.like[(t, c)] = {m: sorted(f for f in s if f not in all_names) for m, s in frags.items()}

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
    """조건 연산자. OR 로 잇는 이름 컬럼은 LIKE 만 (이름 = / IN 은 단독으로만 쓴다), OR 의 항에는 != 를 쓰지 않는다."""
    if conn == "OR" and (table, col) in schema.UNIQUE_KEYS:
        return "LIKE"
    ratio = cfg.in_ratio_unique if (table, col) in schema.UNIQUE_KEYS else cfg.in_ratio
    if schema.in_ok(table, col) and rng.random() < ratio:
        return "IN"
    return rng.choice([op for op in schema.OPS[schema.kind(table, col)] if not (conn == "OR" and op == "!=")])


def _make_cond(rng: random.Random, pools: Pools, table: str, col: str, op: str, cfg: GenConfig
               ) -> tuple[Cond | None, bool]:
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
            c, ne = Cond(first_col, "=", fixed[2]), False
        elif pin:
            c, ne = Cond(first_col, pin[2], pin[3]), False
        else:
            c, ne = _make_cond(rng, pools, table, first_col, cell[2], cfg)
        if c is None:
            return None
        conds.append(c)
        nonexist |= ne
        rest = _pick_cols(rng, table, n - 1, {first_col}, cfg, no_id=conn == "OR")
    else:
        rest = _pick_cols(rng, table, n, set(), cfg, no_id=conn == "OR")
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
    want_order = False if exists else shape["order"] if shape else (n == 0 or rng.random() < cfg.order_ratio)
    if want_order:
        ocands = [c for c in cols if schema.orderable(table, c) and c not in eq_cols]
        if ocands:
            order = (rng.choice(ocands), rng.choice(["ASC", "DESC"]))
            if conn != "OR" and (shape["limit"] if shape else rng.random() < cfg.limit_ratio):
                limit = rng.choices(list(LIMIT_WEIGHTS), weights=list(LIMIT_WEIGHTS.values()))[0]
        elif n == 0 or shape:
            return None  # 정렬이 필요한데 정렬할 컬럼이 없음

    select = conds[0].col if exists else rng.choice(cols + ["*"])
    q = canonical(Query(table, select, tuple(conds), conn, order, limit))
    if violations(q):
        return None

    # 조건 하나만으로 0행(없는 ID 제외)이거나 전체 행이면 의미 없는 조건
    for c in q.conds:
        k = pools.cond_count(table, c)
        is_missing_id = schema.kind(table, c.col) == "id" and k == 0
        if (k == 0 and not is_missing_id) or k == pools.n_rows[table]:
            return None

    n_rows = len(pools.con.execute(q.render()).fetchall())
    if q.conn == "OR" and n_rows == pools.n_rows[table]:
        return None  # OR 로 이어 전체 행이 나오면 의미 없는 조건
    if q.order:
        # 정렬은 2행 이상일 때만, LIMIT 은 실제로 잘라낼 때만 의미가 있다
        base = Query(table, q.select, q.conds, q.conn).render()
        base_rows = len(pools.con.execute(base).fetchall())
        if base_rows < 2 or (q.limit is not None and base_rows <= q.limit):
            return None
    meta = {"n_rows": n_rows, "is_nonexistent": nonexist,
            "has_id": any(schema.kind(table, c.col) == "id" for c in q.conds)}
    return Query(q.table, q.select, q.conds, q.conn, q.order, q.limit, meta)


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
    out = {
        "indist": indist,
        "train": [q for q in rows if q.render() not in indist_sql],
        "struct": [_with_meta(q, struct_combo=i) for i, s in enumerate(struct) for q in s.values()],
        "holdout": generate_holdout(rng, pools, ho, cfg, combos, set(main)),
    }
    return out


def _with_meta(q: Query, **kw) -> Query:
    return Query(q.table, q.select, q.conds, q.conn, q.order, q.limit, {**q.meta, **kw})


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
    return out


HOLDOUT_SHAPES = {
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

    유일 키(이름·PK)는 규칙상 단독(= / IN)만 가능하고, orders 의 FK 는 AND·IN·정렬·LIMIT 까지 시험한다.
    결과 0행, 구조 홀드아웃 조합은 피한다 (값 복사와 구조 일반화를 섞지 않으려고).
    """
    rows: list[Query] = []
    seen = set(seen)
    for tier, table, col, value in holdout_literals(ho):
        unique = (table, col) in schema.UNIQUE_KEYS
        for name in (cfg.holdout_shapes_unique if unique else cfg.holdout_shapes_fk):
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

DEFAULT_OUT = str(stage_paths(2).sql_dir)  # data/stage2/sql (1단계 data/stage1/sql 과 같은 파일 이름)
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
    with open(os.path.join(out_dir, REPORT_FILE), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def _load(out_dir: str) -> tuple[dict, dict[str, list[dict]]]:
    report = json.load(open(os.path.join(out_dir, REPORT_FILE), encoding="utf-8"))
    parts = {name: json.load(open(os.path.join(out_dir, fn), encoding="utf-8")) for name, fn in FILES.items()}
    return report, parts


def _query(d: dict) -> Query:
    meta = {k: d[k] for k in ("n_rows", "is_nonexistent", "has_id") if k in d}
    q = from_dict(d)
    return Query(q.table, q.select, q.conds, q.conn, q.order, q.limit, meta)


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
            if q.order:
                base = len(con.execute(Query(q.table, q.select, q.conds, q.conn).render()).fetchall())
                if base < 2 or (q.limit is not None and base <= q.limit):
                    fails.append(f"[{name}] 의미 없는 ORDER BY/LIMIT {d['sql']!r} (정렬 전 {base}행)")

    # 2. 미등장 리터럴: 학습·분포 내·구조 평가에는 없고, holdout 에는 반드시 있음
    def banned_conds(d: dict) -> list:
        out = []
        for c in d["conds"]:
            b = bans.get(d["table"], {}).get(c["col"], set())
            if c["op"] == "=" and c["val"] in b:
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

    # 8. 결정성
    if check_determinism:
        regen = generate(con, ho, cfg)
        for name, qs in regen.items():
            if [q.render() for q in qs] != sets[name]:
                fails.append(f"[{name}] 같은 설정으로 다시 만든 결과가 저장본과 다름")
    con.close()
    return fails


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------

def summary(out_dir: str) -> None:
    report = json.load(open(os.path.join(out_dir, REPORT_FILE), encoding="utf-8"))
    for name, st in report["stats"].items():
        print(f"[{name}] {st['n']}개, 골격 {st['n_skeletons']}종 | ID {st['id_ratio']:.1%} · "
              f"0행 {st['empty_ratio']:.1%} · 1단계 모양 {st['stage1_ratio']:.1%}")
        for key in ("n_conds", "conn", "order", "limit", "table", "select_kind", "ops"):
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


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="2단계 SQL 생성기")
    ap.add_argument("command", choices=["build", "verify", "summary"])
    p1 = stage_paths(1)
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"SQL 출력 폴더 (기본 {DEFAULT_OUT})")
    ap.add_argument("--db-dir", default=str(p1.db_dir),
                    help=f"shop.db·holdout.json 폴더 (기본 {p1.db_dir}, build 만. verify 는 리포트에 적힌 값)")
    ap.add_argument("--stage1-dir", default=str(p1.data_dir),
                    help=f"겹침 금지용 1단계 쌍 폴더 (train/*.json, eval/<이름>/pairs.json, 기본 {p1.data_dir}, build 만)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=40000, help="학습 SQL 수")
    ap.add_argument("--n-eval-indist", type=int, default=2000, help="분포 내 평가 SQL 수")
    ap.add_argument("--n-eval-struct", type=int, default=2000, help="구조 일반화 평가 SQL 수 (조합 수로 나눔)")
    ap.add_argument("--cell-floor", type=int, default=200, help="(테이블, 컬럼, 연산자) 칸마다 생성 시 강제로 채우는 수")
    ap.add_argument("--cell-floor-train", type=int, default=150, help="verify: 분할 후 학습에 남아야 하는 칸별 수")
    ap.add_argument("--no-determinism", action="store_true", help="verify: 재생성 비교 생략")
    args = ap.parse_args()

    if args.command == "build":
        cfg = GenConfig(seed=args.seed, n_train=args.n_train, n_eval_indist=args.n_eval_indist,
                        n_eval_struct=args.n_eval_struct, cell_floor=args.cell_floor,
                        cell_floor_train=args.cell_floor_train, stage1_data_dir=args.stage1_dir)
        report = build(args.db_dir, args.out, cfg)
        for name, st in report["stats"].items():
            print(f"생성  {os.path.join(args.out, FILES[name])} ({st['n']}개)")
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
