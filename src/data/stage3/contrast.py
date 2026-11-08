# 역할: 학습 SQL의 의미 차이를 드러내는 질문·SQL 대비 묶음을 만든다.
# 사용처: 3단계에서 조건 범위·집계 등 혼동하기 쉬운 의미를 학습 데이터로 보강할 때 사용한다.

"""Build train-only semantic contrasts from configurable training inputs.

SQL variants use the existing AST and training SQL. Questions use the existing
training composer; no evaluation phrases or external question generator is used.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from src.config import stage_paths
from src.data import db_gen
from src.data.json_io import json_files, read_json
from src.data.stage3 import compose, schema, sql_ast, sql_gen
from src.tokenizer.bpe import Tokenizer

PATHS = stage_paths(3)


def read(path):
    return read_json(path)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(sql):
    return sql_ast.canonical(sql_ast.parse(sql)).render()


def banned_values(q, bans):
    for scope in ("conds", "inner_conds"):
        for c in getattr(q, scope, ()):
            values = c.val if c.op == "IN" else (c.val, c.val2) if c.op == "BETWEEN" else (c.val,)
            banned = bans.get(q.table, {}).get(c.col, set())
            if any((v.strip("%") if c.op == "LIKE" else v) in banned for v in values):
                return True
    return False


def evaluation_sql():
    paths = sorted(PATHS.data_dir.glob("eval*/**/sql.json"))
    paths += sorted(stage_paths(1).eval_dir.glob("**/sql.json"))
    sqls = set()
    for path in paths:
        for row in read(path):
            try:
                sqls.add(identity(row["sql"]))
            except (ValueError, KeyError):
                # Old-stage unsupported SQL is irrelevant to this AST.
                continue
    return sqls, paths


def variants(base, kind, rng):
    """Return variants differing only in projection, one operator, order, or LIKE mode."""
    q = replace(base, meta={}, policy_order=True, tie_break=None)
    if kind == "projection":
        cols = [c for c in schema.TABLE_COLUMNS[q.table]
                if not any(x.col == c and x.op in ("=", "IN") for x in q.conds)]
        if len(cols) < 2:
            return []
        return [replace(q, select=c, order=None, limit=None) for c in rng.sample(cols, 2)]
    if kind == "comparison":
        eligible = [i for i, c in enumerate(q.conds)
                    if schema.kind(q.table, c.col) in ("num", "date")
                    and c.op in ("=", "<", ">", "<=", ">=")]
        if not eligible:
            return []
        idx = rng.choice(eligible)
        result = []
        for op in ("=", "<", ">", "<=", ">="):
            conds = list(q.conds)
            conds[idx] = replace(conds[idx], op=op)
            result.append(replace(q, conds=tuple(conds), order=None, limit=None))
        return result
    if kind == "ordering":
        fixed = {c.col for c in q.conds if c.op in ("=", "IN")}
        cols = [c for c in schema.TABLE_COLUMNS[q.table]
                if schema.orderable(q.table, c) and c not in fixed]
        if len(cols) < 2:
            return []
        a, b = rng.sample(cols, 2)
        n, m = rng.sample((1, 3, 5, 10), 2)
        return [replace(q, order=order, limit=limit)
                for order, limit in (((a, "ASC"), n), ((a, "DESC"), n),
                                     ((b, "ASC"), n), ((a, "ASC"), m))]
    if kind == "like":
        eligible = [i for i, c in enumerate(q.conds) if c.op == "LIKE"]
        if len(eligible) != 1:
            return []
        idx = eligible[0]
        fragment = q.conds[idx].val.strip("%")
        result = []
        for pattern in (fragment + "%", "%" + fragment, "%" + fragment + "%"):
            conds = list(q.conds)
            conds[idx] = replace(conds[idx], val=pattern)
            result.append(replace(q, conds=tuple(conds), order=None, limit=None))
        return result
    raise ValueError(kind)


def build(out, quotas, seed, db, *, corpus_files=None, round_name=None, source_sql=None):
    out = Path(out)
    if (out / "pairs_checked.json").exists():
        raise FileExistsError("contrast round already exists")
    rng = random.Random(seed)
    source_path = Path(source_sql) if source_sql is not None else PATHS.sql_dir / "train.json"
    sources = [sql_ast.from_dict(r) for r in read(source_path)]
    sources = [q for q in sources if isinstance(q, sql_ast.Query)]
    pools = {kind: [q for q in sources if kind != "like" or any(c.op == "LIKE" for c in q.conds)]
             for kind in quotas}
    excluded, eval_paths = evaluation_sql()
    bans = db_gen.load_bans(read(PATHS.holdout_path))
    combos = sql_gen.combos_of(sql_gen.GenConfig())
    comp = compose.Composer.load("train")
    tok = Tokenizer.load(PATHS.tokenizer_path)
    corpus_files = json_files(PATHS.train_dir) if corpus_files is None else corpus_files
    corpus = [p for f in corpus_files for p in read(f)]
    seen_questions = {p["question"] for p in corpus}
    used_sql = set()
    result, manifests = [], []
    rejected = Counter()
    con = sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)
    try:
        for kind, n_groups in quotas.items():
            accepted = 0
            for attempt in range(max(1000, n_groups * 300)):
                if accepted == n_groups:
                    break
                queries = [sql_ast.canonical(q) for q in variants(rng.choice(pools[kind]), kind, rng)]
                if not queries:
                    rejected["no_variants"] += 1
                    continue
                keys = [q.render() for q in queries]
                if len(set(keys)) != len(keys) or set(keys) & (excluded | used_sql):
                    rejected["sql_overlap"] += 1
                    continue
                if any(sql_ast.violations(q) or sql_gen.struct_hit(q, combos) is not None
                       or banned_values(q, bans) for q in queries):
                    rejected["scope_or_holdout"] += 1
                    continue
                gid = f"{kind}_{accepted + 1:04}"
                try:
                    members = comp.compose_contrast(queries, seed + attempt, gid)
                except compose.ComposeError:
                    rejected["composition"] += 1
                    continue
                candidate = []
                for arm, member in enumerate(members):
                    for style, pair in enumerate(member):
                        candidate.append({**pair, "source": "composer_contrast", "round": round_name or out.name,
                                          "contrast_group": gid, "contrast_kind": kind,
                                          "contrast_role": str(arm), "style": style})
                questions = [p["question"] for p in candidate]
                if len(set(questions)) != len(questions) or set(questions) & seen_questions:
                    rejected["question_overlap"] += 1
                    continue
                if any(len(tok.encode(p["question"])) + len(tok.encode(p["sql"])) + 3 > 256
                       or len(tok.encode(p["sql"])) + 1 > 128 for p in candidate):
                    rejected["length"] += 1
                    continue
                # Require at least one nonempty arm and at least two different
                # observable results; changes must matter on the actual DB.
                outputs = [con.execute(q.render()).fetchall() for q in queries]
                if not any(outputs) or len({repr(o) for o in outputs}) < 2:
                    rejected["unobservable"] += 1
                    continue
                for pair in candidate:
                    if sql_ast.parse(pair["sql"]).render() != pair["sql"]:
                        raise ValueError("SQL roundtrip failure")
                    if tok.decode(tok.encode(pair["question"])) != pair["question"]:
                        raise ValueError("tokenizer roundtrip failure")
                result.extend(candidate)
                used_sql.update(keys)
                seen_questions.update(questions)
                manifests.append({"group": gid, "kind": kind, "sql": keys,
                                  "pair_count": len(candidate), "result_counts": list(map(len, outputs))})
                accepted += 1
            else:
                raise RuntimeError(f"could not fill {kind}: {accepted}/{n_groups}")
            print(f"{kind}: {accepted} groups", flush=True)
    finally:
        con.close()
    save(out / "pairs_checked.json", result)
    save(out / "groups.json", manifests)
    protected = [*eval_paths, PATHS.tokenizer_path, PATHS.holdout_path,
                 Path("docs/text-to-sql-label-rules.md")]
    save(out / "report.json", {"seed": seed, "quotas": quotas, "pairs": len(result),
                               "unique_sql": len(used_sql), "rejected": rejected,
                               "by_kind": Counter(p["contrast_kind"] for p in result),
                               "source_sha256": {str(source_path): sha(source_path)},
                               "protected_sha256": {str(p): sha(p) for p in protected}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-sql", type=Path, default=PATHS.sql_dir / "train.json")
    parser.add_argument("--corpus-files", nargs="+", type=Path,
                        help="Training question files checked for duplicate questions")
    parser.add_argument("--round", dest="round_name")
    parser.add_argument("--projection-groups", type=int, default=200)
    parser.add_argument("--comparison-groups", type=int, default=80)
    parser.add_argument("--ordering-groups", type=int, default=100)
    parser.add_argument("--like-groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db", type=Path, default=PATHS.db_path)
    args = parser.parse_args()
    quotas = {kind: getattr(args, kind + "_groups")
              for kind in ("projection", "comparison", "ordering", "like")}
    if any(n < 0 for n in quotas.values()) or not any(quotas.values()):
        parser.error("group counts must be nonnegative with at least one positive count")
    build(args.out, {kind: n for kind, n in quotas.items() if n}, args.seed, args.db,
          corpus_files=args.corpus_files, round_name=args.round_name, source_sql=args.source_sql)


if __name__ == "__main__":
    main()
