# 역할: 3단계 평가 질문·정답 쌍과 SQL 메타데이터를 구성한다.
# 사용처: 3단계 평가 입력을 data/stage3/eval/에 준비할 때 사용한다.

"""Build stage-3 composer evaluation questions; no external/human sets implied."""
import argparse
from pathlib import Path
from src.data.stage3.compose import (
    Composer, ComposeError, question_order_sql, rng_for,
    FRAMES_PATH, LEXICON_PATH, SYNONYMS_PATH, AGGREGATE_TEMPLATES_PATH,
    _load_records,
)
from src.data.stage3.clean_ambiguous import drop_reason
from src.evaluation.eval_sets import write_combined
from src.config import stage_paths
PATHS = stage_paths(3)
COMPOSER_SETS = ("indist", "struct", "holdout")
PHRASE_SOURCE = "indist"
COMPOSER_SEED = 1

def _sql_file(split: str, sql_dir: Path) -> Path:
    return sql_dir / f"eval_{split}.json"


def _pair(question: str, rec: dict, **extra) -> dict:
    q = question.strip().lower()
    try:
        sql = question_order_sql(rec, q)
    except ComposeError:  # IN 값을 질문에서 못 찾으면 정규형 그대로 (채점은 IN 값 순서를 무시한다)
        sql = rec["sql"]
    return {"question": q, "sql": sql, "sql_group": rec["sql"], **extra}


def build_composer(sql_dir: Path, seed: int = COMPOSER_SEED, **composer_paths) -> dict[str, list[dict]]:
    comp = Composer.load("train", **composer_paths)
    records = _load_records([_sql_file(split, sql_dir) for split in COMPOSER_SETS])
    sets = {}
    for split in COMPOSER_SETS:
        rows = records[f"eval_{split}"]
        pairs, failed = [], 0
        for r in rows:
            try:
                pairs.append(_pair(comp.compose(r, rng_for(seed, 0, r["sql"]))[0], r))
            except ComposeError:
                failed += 1
        if failed:
            print(f"  [{split}] 조합 실패 {failed}개 제외")
        sets[split] = pairs

    rows = records[f"eval_{PHRASE_SOURCE}"]
    pairs = []
    for r in rows:
        res = comp.compose_phrase(r, seed, 0)
        if res is not None:
            q, _, _, fam = res
            pairs.append(_pair(q, r, eval_slot=fam))
    print(f"  [phrase] eval 자리가 없는 SQL {len(rows) - len(pairs)}개 제외")
    sets["phrase"] = pairs
    if "advanced_indist" in records:
        for split in COMPOSER_SETS:
            rows = records[f"advanced_{split}"]
            sets[f"advanced_{split}"] = [
                _pair(comp.compose(r, rng_for(seed, 0, r["sql"]))[0], r, source="composer_advanced")
                for r in rows
            ]
        evaluator = Composer.load("eval", **composer_paths)
        rows = records["advanced_indist"]
        sets["advanced_phrase"] = [
            _pair(evaluator.compose(r, rng_for(seed, 0, r["sql"]))[0], r, source="composer_advanced")
            for r in rows
        ]
    # 라벨 판정(clean_ambiguous)에 걸린 문항은 버린다. 검사기 단서는 학습 문장으로만 정하므로, 평가 전용 표현에서
    # 판정이 어긋나도 검사기를 평가 문장에 맞추지 않고 그 문항을 뺀다
    for name, pairs in sets.items():
        kept = [p for p in pairs if not drop_reason(p)]
        if len(kept) != len(pairs):
            print(f"  [{name}] 라벨 판정에 걸린 {len(pairs) - len(kept)}개 제외")
        sets[name] = kept
    return sets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["composer"], default="composer")
    parser.add_argument("--sql-dir", default=str(PATHS.sql_dir))
    parser.add_argument("--eval-dir", "--out", dest="eval_dir", default=str(PATHS.eval_dir))
    parser.add_argument("--seed", type=int, default=COMPOSER_SEED)
    parser.add_argument("--frames", default=str(FRAMES_PATH))
    parser.add_argument("--lexicon", default=str(LEXICON_PATH))
    parser.add_argument("--synonyms", default=str(SYNONYMS_PATH))
    parser.add_argument("--templates", default=str(AGGREGATE_TEMPLATES_PATH))
    args = parser.parse_args(argv)
    sql_dir = Path(args.sql_dir)
    sets = build_composer(sql_dir, args.seed, frames_path=args.frames, lexicon_path=args.lexicon,
                          synonyms_path=args.synonyms or None, templates_path=args.templates)
    paths = [_sql_file(s, sql_dir) for s in COMPOSER_SETS]
    write_combined(sets, paths, Path(args.eval_dir))
    for name, pairs in sets.items():
        print(name, len(pairs))

if __name__ == "__main__":
    main()
