"""
Text-to-SQL 1단계 SQL 생성기

db_gen.py 가 만든 shop.db / holdout.json 을 입력으로 받아, 1단계 문법
(SELECT 단일 컬럼 또는 *, WHERE 단일 등호)의 SQL을 (테이블 × WHERE 컬럼) 조합 단위로
층화 샘플링해 생성한다.

완료 조건 (계획서 3-3 "1단계 세부 진행 순서" 2번)
  1. 전체 SQL이 SQLite에서 오류 없이 실행
  2. 구조별 할당량이 표로 재현 가능
  3. ID 조건 비중 30% 이하

설계 메모
  - 홀드아웃 리터럴(data/holdout.json)은 db_gen.load_bans() 로 걸러 절대 사용하지 않는다.
  - ID 컬럼(customer_id, item_id, order_id)은 값 종류가 많아 균등 샘플링하면 전체의
    60%+ 를 차지한다(계획서 5-3). cap_id 를 자동으로 줄여 나가며 30% 이하로 맞춘다.
  - 5-2가 지적한 "ID 첫 자리로 테이블 맞히기" 지름길을 깨기 위해, ID 조건 중 일부는
    존재하지 않는 ID(결과 0행)로 채운다 (기본 20%, id_nonexistent_ratio).
  - "분포 내 평가"(3-5)는 값 자체가 아니라 SQL 문자열 단위로 학습/평가를 분리한다.
    (값은 겹칠 수 있지만, 그 SQL 문자열 자체는 학습에 없었던 것이어야 한다는 뜻이므로.)
    값 자체를 완전히 분리하는 건 홀드아웃(미등장 값)의 몫이다.

사용법
    python sql_gen.py build                       # data/sql_train.json, data/sql_eval_indist.json
    python sql_gen.py build --preset small --seed 1 --n-train-total 200   # 파일럿
    python sql_gen.py verify                       # 완료 조건 3종 재검사
    python sql_gen.py test                         # 결정성 · 홀드아웃 미포함 · train/eval 비중복
    python sql_gen.py summary                      # 조합별 할당량 표 출력
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass

import db_gen


# ---------------------------------------------------------------------------
# 컬럼 메타데이터
# ---------------------------------------------------------------------------

COLUMN_TYPES: dict[tuple[str, str], str] = {
    ("customers", "customer_id"): "int",
    ("customers", "name"): "text",
    ("customers", "city"): "text",
    ("customers", "membership"): "text",
    ("items", "item_id"): "int",
    ("items", "item_name"): "text",
    ("items", "category"): "text",
    ("items", "price"): "float",
    ("items", "stock"): "int",
    ("orders", "order_id"): "int",
    ("orders", "customer_id"): "int",
    ("orders", "item_id"): "int",
    ("orders", "quantity"): "int",
    ("orders", "status"): "text",
}

# WHERE 컬럼이 ID 컬럼인지 (5-2/5-3의 "ID 조건" 정의)
ID_COLUMNS = {"customer_id", "item_id", "order_id"}

# ID 가 아니면서 카디널리티가 낮아 "전량 사용" 하는 컬럼 (city, membership, category, quantity, status)
LOW_CARD_COLUMNS = {"city", "membership", "category", "quantity", "status"}

# ID 컬럼의 값 도메인이 어느 테이블 PK 에서 나오는지: "존재하는 값" 판정 기준
ID_DOMAIN_TABLE = {"customer_id": "customers", "item_id": "items", "order_id": "orders"}
ID_RANGE_ATTR = {"customer_id": "customer_id_range", "item_id": "item_id_range",
                  "order_id": "order_id_range"}


@dataclass
class SqlGenConfig:
    seed: int = 0
    preset: str = "large"
    # 비-ID 고카디널리티 컬럼(name, item_name, price, stock)에서 쓸 리터럴 수 상한
    cap_nonid: int = 40
    # ID 컬럼(customer_id, item_id, order_id)에서 쓸 "실존" 리터럴 수 상한
    # (id_ratio_max 를 넘기면 자동으로 줄어든다)
    cap_id: int = 40
    # ID 조건 중 결과 0행(존재하지 않는 id)의 비율 (계획서 5-3)
    id_nonexistent_ratio: float = 0.20
    # 전체 SQL 중 ID 조건이 차지할 수 있는 최대 비율 (계획서 3-3 완료 조건)
    id_ratio_max: float = 0.30
    # SQL 문자열을 학습 : 분포 내 평가 로 나누는 비율
    train_ratio: float = 0.80


# ---------------------------------------------------------------------------
# 리터럴 수집
# ---------------------------------------------------------------------------

def available_literals(con: sqlite3.Connection, bans: dict, table: str, col: str) -> list:
    """홀드아웃으로 금지된 값을 뺀, 결정적으로 정렬된 사용 가능 리터럴 목록."""
    rows = {r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM {table}")}
    rows -= bans.get(table, {}).get(col, set())
    return sorted(rows)


def existing_ids_by_domain(con: sqlite3.Connection) -> dict[str, set[int]]:
    return {
        "customer_id": {r[0] for r in con.execute("SELECT customer_id FROM customers")},
        "item_id": {r[0] for r in con.execute("SELECT item_id FROM items")},
        "order_id": {r[0] for r in con.execute("SELECT order_id FROM orders")},
    }


# ---------------------------------------------------------------------------
# 할당량 결정 (테이블 × WHERE 컬럼 단위)
# ---------------------------------------------------------------------------

def _select_variants(table: str) -> list[str]:
    return db_gen.TABLE_COLUMNS[table] + ["*"]


def decide_quota_pools(con: sqlite3.Connection, bans: dict, cfg: SqlGenConfig
                       ) -> dict[tuple[str, str], dict]:
    """(table, col) -> {"exist_pool": [...], "nonexistent_pool": [...]}.

    풀은 seed 가 같으면 항상 같은 순서로 섞인다 (재현성). cap_id 는 나중에
    apply_id_ratio_cap() 이 이 풀의 앞부분만 잘라 쓰는 방식으로 줄인다.
    """
    rng = random.Random(cfg.seed)
    dbcfg = db_gen.PRESETS[cfg.preset]
    existing = existing_ids_by_domain(con)

    pools: dict[tuple[str, str], dict] = {}
    for table, cols in db_gen.TABLE_COLUMNS.items():
        for col in cols:
            avail = available_literals(con, bans, table, col)
            if col in ID_COLUMNS:
                exist_pool = rng.sample(avail, len(avail))  # 섞기만 함(전량은 cap_id 가 자름)
                lo, hi = getattr(dbcfg, ID_RANGE_ATTR[col])
                domain_existing = existing[col]
                candidates = [v for v in range(lo, hi + 1) if v not in domain_existing]
                nonexistent_pool = rng.sample(candidates, len(candidates))
            else:
                exist_pool = rng.sample(avail, len(avail))
                nonexistent_pool = []
            pools[(table, col)] = {"exist_pool": exist_pool, "nonexistent_pool": nonexistent_pool}
    return pools


def _combo_counts(pools: dict, cfg: SqlGenConfig, cap_id: int) -> dict[tuple[str, str], dict]:
    """주어진 cap_id 로 각 (table, col) 조합의 실제 사용 리터럴 수 / SQL 수를 계산."""
    counts = {}
    for (table, col), pool in pools.items():
        n_variants = len(_select_variants(table))
        if col in LOW_CARD_COLUMNS:
            n_exist = len(pool["exist_pool"])
            n_non = 0
        elif col in ID_COLUMNS:
            n_exist = min(cap_id, len(pool["exist_pool"]))
            n_non = round(n_exist * cfg.id_nonexistent_ratio / (1 - cfg.id_nonexistent_ratio))
            n_non = min(n_non, len(pool["nonexistent_pool"]))
        else:
            n_exist = min(cfg.cap_nonid, len(pool["exist_pool"]))
            n_non = 0
        n_sql = (n_exist + n_non) * n_variants
        counts[(table, col)] = {
            "n_variants": n_variants, "n_exist": n_exist, "n_nonexistent": n_non, "n_sql": n_sql,
            "is_id": col in ID_COLUMNS,
        }
    return counts


def apply_id_ratio_cap(pools: dict, cfg: SqlGenConfig) -> tuple[dict, int]:
    """id_ratio 가 id_ratio_max 를 넘으면 cap_id 를 줄여가며 맞춘다."""
    cap_id = cfg.cap_id
    while cap_id > 1:
        counts = _combo_counts(pools, cfg, cap_id)
        total = sum(c["n_sql"] for c in counts.values())
        id_total = sum(c["n_sql"] for c in counts.values() if c["is_id"])
        if total == 0 or id_total / total <= cfg.id_ratio_max:
            return counts, cap_id
        cap_id -= 1
    return _combo_counts(pools, cfg, cap_id), cap_id


# ---------------------------------------------------------------------------
# SQL 문자열 생성
# ---------------------------------------------------------------------------

def format_literal(value, coltype: str) -> str:
    if coltype == "text":
        return f"'{value}'"
    if coltype == "float":
        return f"{value:.2f}"
    return str(value)


def build_sql(table: str, select_col: str, where_col: str, value) -> str:
    coltype = COLUMN_TYPES[(table, where_col)]
    select_clause = "*" if select_col == "*" else select_col
    return f"SELECT {select_clause} FROM {table} WHERE {where_col} = {format_literal(value, coltype)}"


def generate(con: sqlite3.Connection, bans: dict, cfg: SqlGenConfig
            ) -> tuple[list[dict], list[dict], list[dict]]:
    """(train, eval, quota_table) 반환. quota_table 은 요약 출력/재현성 검증용."""
    pools = decide_quota_pools(con, bans, cfg)
    counts, cap_id_used = apply_id_ratio_cap(pools, cfg)

    rng = random.Random(cfg.seed ^ 0x5151)  # SQL 리스트 셔플용 (풀 셔플과 분리)
    train: list[dict] = []
    ev: list[dict] = []
    quota_table: list[dict] = []

    for (table, col), pool in pools.items():
        c = counts[(table, col)]
        literals = [(v, False) for v in pool["exist_pool"][:c["n_exist"]]]
        literals += [(v, True) for v in pool["nonexistent_pool"][:c["n_nonexistent"]]]

        combo_sqls = []
        for value, is_nonexistent in literals:
            for sel in _select_variants(table):
                sql = build_sql(table, sel, col, value)
                n_rows = con.execute(sql).fetchall()
                combo_sqls.append({
                    "sql": sql, "table": table, "select": sel, "where_col": col,
                    "where_val": value, "is_id_condition": col in ID_COLUMNS,
                    "is_nonexistent": is_nonexistent, "n_rows": len(n_rows),
                })

        rng.shuffle(combo_sqls)
        split = round(len(combo_sqls) * cfg.train_ratio)
        train.extend(combo_sqls[:split])
        ev.extend(combo_sqls[split:])

        quota_table.append({
            "table": table, "where_col": col, "is_id": col in ID_COLUMNS,
            "n_exist": c["n_exist"], "n_nonexistent": c["n_nonexistent"],
            "n_select_variants": c["n_variants"], "n_sql": c["n_sql"],
            "n_train": split, "n_eval": len(combo_sqls) - split,
        })

    quota_table.sort(key=lambda r: (r["table"], r["where_col"]))
    return train, ev, quota_table


# ---------------------------------------------------------------------------
# 빌드 / 저장
# ---------------------------------------------------------------------------

def _paths(out_dir: str) -> dict[str, str]:
    return {
        "db": os.path.join(out_dir, "shop.db"),
        "holdout": os.path.join(out_dir, "holdout.json"),
        "train": os.path.join(out_dir, "sql_train.json"),
        "eval": os.path.join(out_dir, "sql_eval_indist.json"),
        "report": os.path.join(out_dir, "sql_gen_report.json"),
    }


def build(out_dir: str, cfg: SqlGenConfig) -> dict:
    paths = _paths(out_dir)
    con = sqlite3.connect(paths["db"])
    bans = db_gen.load_bans(json.load(open(paths["holdout"], encoding="utf-8")))

    train, ev, quota_table = generate(con, bans, cfg)
    con.close()

    total = sum(q["n_sql"] for q in quota_table)
    id_total = sum(q["n_sql"] for q in quota_table if q["is_id"])
    report = {
        "config": asdict(cfg),
        "quota_table": quota_table,
        "n_train": len(train), "n_eval": len(ev), "n_total_sql": total,
        "id_ratio": round(id_total / total, 4) if total else 0.0,
    }

    with open(paths["train"], "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(paths["eval"], "w", encoding="utf-8") as f:
        json.dump(ev, f, ensure_ascii=False, indent=2)
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def verify(out_dir: str) -> list[str]:
    """완료 조건 3종: 실행 오류 없음 / 할당량 표 재현 가능 / ID 비중 30% 이하."""
    paths = _paths(out_dir)
    fails: list[str] = []
    if not os.path.exists(paths["report"]):
        return [f"{paths['report']} 없음 - 먼저 build 를 실행하세요"]

    report = json.load(open(paths["report"], encoding="utf-8"))
    cfg = SqlGenConfig(**report["config"])

    con = sqlite3.connect(paths["db"])
    bans = db_gen.load_bans(json.load(open(paths["holdout"], encoding="utf-8")))
    train, ev, quota_table = generate(con, bans, cfg)

    # 1. 전체 SQL이 SQLite에서 오류 없이 실행 (+ 저장된 n_rows 와 일치)
    for row in train + ev:
        try:
            n = len(con.execute(row["sql"]).fetchall())
        except sqlite3.Error as e:
            fails.append(f"실행 오류: {row['sql']!r} ({e})")
            continue
        if n != row["n_rows"]:
            fails.append(f"n_rows 불일치: {row['sql']!r} 저장값 {row['n_rows']} != 실제 {n}")
    con.close()

    # 2. 구조별 할당량이 표로 재현 가능 (같은 config 로 다시 만들면 같은 표가 나와야 함)
    if quota_table != report["quota_table"]:
        fails.append("quota_table 이 재현되지 않음 (config 로부터 다시 만든 표가 저장된 표와 다름)")

    # 3. ID 조건 비중 30% 이하
    total = sum(q["n_sql"] for q in quota_table)
    id_total = sum(q["n_sql"] for q in quota_table if q["is_id"])
    id_ratio = id_total / total if total else 0.0
    if id_ratio > cfg.id_ratio_max + 1e-9:
        fails.append(f"ID 조건 비중 {id_ratio:.1%} > 상한 {cfg.id_ratio_max:.0%}")

    return fails


def canonical_hash(rows: list[dict]) -> str:
    keys = [(r["sql"],) for r in rows]
    return hashlib.sha256(json.dumps(sorted(keys), sort_keys=True).encode()).hexdigest()


def test(out_dir: str, cfg: SqlGenConfig) -> list[tuple[str, list[str]]]:
    paths = _paths(out_dir)
    con = sqlite3.connect(paths["db"])
    ho = json.load(open(paths["holdout"], encoding="utf-8"))
    bans = db_gen.load_bans(ho)

    train_a, eval_a, _ = generate(con, bans, cfg)
    train_b, eval_b, _ = generate(con, bans, cfg)
    con.close()

    results = []

    f = []
    if canonical_hash(train_a) != canonical_hash(train_b) or canonical_hash(eval_a) != canonical_hash(eval_b):
        f.append("같은 seed 인데 생성 결과가 다름")
    results.append(("결정성", f))

    f = []
    for row in train_a + eval_a:
        banned = bans.get(row["table"], {}).get(row["where_col"], set())
        if row["where_val"] in banned:
            f.append(f"홀드아웃 리터럴 사용됨: {row['sql']!r}")
    results.append(("홀드아웃 미포함", f))

    f = []
    train_sqls = {r["sql"] for r in train_a}
    eval_sqls = {r["sql"] for r in eval_a}
    overlap = train_sqls & eval_sqls
    if overlap:
        f.append(f"train/eval SQL 중복 {len(overlap)}건 (예: {next(iter(overlap))!r})")
    results.append(("train/eval 비중복", f))

    f = []
    total = len(train_a) + len(eval_a)
    id_total = sum(1 for r in train_a + eval_a if r["is_id_condition"])
    ratio = id_total / total if total else 0.0
    if ratio > cfg.id_ratio_max + 1e-9:
        f.append(f"ID 조건 비중 {ratio:.1%} > 상한 {cfg.id_ratio_max:.0%}")
    results.append(("ID 조건 비중", f))

    return results


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------

def summary(out_dir: str) -> None:
    paths = _paths(out_dir)
    if not os.path.exists(paths["report"]):
        print(f"{paths['report']} 없음 - 먼저 build 를 실행하세요")
        return
    report = json.load(open(paths["report"], encoding="utf-8"))

    print(f"{'테이블':<11}{'WHERE 컬럼':<14}{'실존':>6}{'미존재':>8}{'변형':>6}{'SQL':>8}{'train':>8}{'eval':>7}")
    print("-" * 70)
    for q in report["quota_table"]:
        tag = "ID" if q["is_id"] else ""
        print(f"{q['table']:<11}{q['where_col']:<14}{q['n_exist']:>6}{q['n_nonexistent']:>8}"
              f"{q['n_select_variants']:>6}{q['n_sql']:>8}{q['n_train']:>8}{q['n_eval']:>7}  {tag}")
    print("-" * 70)
    print(f"{'합계':<25}{'':>19}{report['n_total_sql']:>8}{report['n_train']:>8}{report['n_eval']:>7}")
    print(f"ID 조건 비중  {report['id_ratio']:.1%}  (상한 {report['config']['id_ratio_max']:.0%})")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="1단계 SQL 생성기")
    ap.add_argument("command", choices=["build", "verify", "test", "summary"])
    ap.add_argument("--preset", default="large", choices=list(db_gen.PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--cap-nonid", type=int, default=40)
    ap.add_argument("--cap-id", type=int, default=40)
    ap.add_argument("--id-ratio-max", type=float, default=0.30)
    ap.add_argument("--train-ratio", type=float, default=0.80)
    args = ap.parse_args()

    cfg = SqlGenConfig(seed=args.seed, preset=args.preset, cap_nonid=args.cap_nonid,
                       cap_id=args.cap_id, id_ratio_max=args.id_ratio_max,
                       train_ratio=args.train_ratio)

    if args.command == "build":
        report = build(args.out, cfg)
        print(f"생성  {os.path.join(args.out, 'sql_train.json')} ({report['n_train']}개)")
        print(f"      {os.path.join(args.out, 'sql_eval_indist.json')} ({report['n_eval']}개)")
        fails = verify(args.out)
        print("검증 ", "통과" if not fails else "실패\n  - " + "\n  - ".join(fails))
        print(f"ID 조건 비중  {report['id_ratio']:.1%}")
        return 1 if fails else 0

    if args.command == "verify":
        fails = verify(args.out)
        print("통과" if not fails else "실패\n  - " + "\n  - ".join(fails))
        return 1 if fails else 0

    if args.command == "test":
        for name, fails in test(args.out, cfg):
            print(f"[{'PASS' if not fails else 'FAIL'}] {name}")
            for x in fails:
                print(f"       {x}")
        return 0

    summary(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
