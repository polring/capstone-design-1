"""
eval_build.py — 2단계 평가셋 만들기 (data/stage2/eval/pairs.json + sql.json, 합친 방식)

모든 평가셋을 파일 하나에 담고 쌍마다 set(평가셋 이름)을 적는다(src/evaluation/eval_sets.py). 쌍은 question(소문자, 추론과 같게),
sql(IN 값은 질문에 나온 순서, L8), sql_group(정규형 SQL, sql.json 키)과 출처 정보(persona, split, writer, eval_slot …)를 가진다.
sql.json 은 sql_gen 평가 분할 출력에서 쓰인 SQL 만 뽑아 만든다(eval_sets.write_combined). 명령마다 그 평가셋만 바꾸고 나머지는
그대로 둔다. 다시 실행하면 같은 결과가 나온다(조합기 문장은 seed 로 정해진다).

    python -m src.data.stage2.eval_build              # 아래 전부
    python -m src.data.stage2.eval_build composer     # indist / struct / holdout / phrase: 평가 SQL 의 조합기 문장
    python -m src.data.stage2.eval_build ext          # ext_<라운드>_<플랫폼>: 외부 말투 평가 라운드의 검증 통과 쌍
    python -m src.data.stage2.eval_build human        # human: 사람 작성 세트에서 질문을 채운 문항

- composer: indist / struct / holdout 은 train 조합기로 SQL 마다 문장 하나(학습 문장과 다른 seed)다. 처음 보는 SQL·구조·값을 잰다.
  phrase 는 indist SQL 에서 eval 자리 하나(틀 / 조건 키 / 정렬 키)만 eval 표현으로 쓴 문장이다(처음 보는 표현).
- ext: data_raw/stage2/ext_*/pairs_<플랫폼>.json (question_gen validate 결과). 라벨 규칙과 어긋난 문항은 validate 가 이미 뺐다.
  dev / 봉인 구분은 2단계 보고서에 둔다(이름에는 넣지 않는다). --ext-rounds 없이(라운드 전부) 실행하면 이번에 만들지 않은
  ext_* 평가셋(응답 폴더를 지운 플랫폼 등)은 지운다.
- human: data_raw/stage2/human/sql_list.json(문항 목록)과 작성자 파일 sql_list_<작성자>.json(또는 .csv)의 question.
  질문이 빈 문항은 건너뛰고, 값 검사(question_gen.check_question)에 걸린 문항은 정답이 질문과 맞지 않으므로 빼고 출력한다.
  라벨 규칙 판정은 빼지 않는다(규칙 일치 EM 이 따로 뺀다). 팀원 질문이 오면 다시 실행하면 된다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import stage_paths
from src.data.stage2 import question_gen
from src.data.stage2.compose import ComposeError, Composer, question_order_sql, rng_for
from src.evaluation.eval_sets import SET_KEY, write_combined

PATHS = stage_paths(2)  # 2단계 평가셋이라 config.STAGE 와 무관하게 2단계 경로를 쓴다
COMPOSER_SETS = ("indist", "struct", "holdout")
PHRASE_SOURCE = "indist"
HUMAN_DIR = PATHS.raw_dir / "human"
COMPOSER_SEED = 1  # 학습 문장(compose build)은 seed 0


def _sql_file(split: str, sql_dir: Path) -> Path:
    return sql_dir / f"eval_{split}.json"


def _pair(question: str, rec: dict, **extra) -> dict:
    q = question.strip().lower()
    try:
        sql = question_order_sql(rec, q)
    except ComposeError:  # IN 값을 질문에서 못 찾으면 정규형 그대로 (채점은 IN 값 순서를 무시한다)
        sql = rec["sql"]
    return {"question": q, "sql": sql, "sql_group": rec["sql"], **extra}


def build_composer(sql_dir: Path, seed: int = COMPOSER_SEED) -> dict[str, list[dict]]:
    comp = Composer.load("train")
    sets = {}
    for split in COMPOSER_SETS:
        rows = json.loads(_sql_file(split, sql_dir).read_text(encoding="utf-8"))
        pairs, failed = [], 0
        for r in rows:
            try:
                pairs.append(_pair(comp.compose(r, rng_for(seed, 0, r["sql"]))[0], r))
            except ComposeError:
                failed += 1
        if failed:
            print(f"  [{split}] 조합 실패 {failed}개 제외")
        sets[split] = pairs

    rows = json.loads(_sql_file(PHRASE_SOURCE, sql_dir).read_text(encoding="utf-8"))
    pairs = []
    for r in rows:
        res = comp.compose_phrase(r, seed, 0)
        if res is not None:
            q, _, _, fam = res
            pairs.append(_pair(q, r, eval_slot=fam))
    print(f"  [phrase] eval 자리가 없는 SQL {len(rows) - len(pairs)}개 제외")
    sets["phrase"] = pairs
    return sets


def build_ext(rounds: list[Path]) -> dict[str, list[dict]]:
    keep = ("persona", "platform", "split")
    sets = {}
    for rd in rounds:
        split = json.loads((rd / question_gen.ROUND_META).read_text(encoding="utf-8"))["split"] \
            if (rd / question_gen.ROUND_META).exists() else rd.name.rsplit("_", 1)[-1]
        if split == "train":
            continue  # 학습 SQL 라운드는 평가셋이 아니다
        for pf in sorted(rd.glob("pairs_*.json")):
            platform = pf.stem.removeprefix("pairs_")
            src = json.loads(pf.read_text(encoding="utf-8"))
            pairs = [{"question": p["question"].strip().lower(), "sql": p["sql"], "sql_group": p["sql_group"],
                      **{k: p[k] for k in keep if k in p}} for p in src]
            sets[f"{rd.name}_{platform}"] = pairs
    return sets


def _writer_questions(human_dir: Path) -> dict[int, str]:
    """작성자 파일(sql_list_<작성자>.json, 없으면 .csv)의 id → question. JSON 과 CSV 가 둘 다 있으면 JSON 을 쓴다."""
    out: dict[int, str] = {}
    for jf in sorted(human_dir.glob("sql_list_*.json")):
        for r in json.loads(jf.read_text(encoding="utf-8")):
            if str(r.get("question", "")).strip():
                out[int(r["id"])] = r["question"]
    for cf in sorted(human_dir.glob("sql_list_*.csv")):
        if cf.with_suffix(".json").exists():
            continue
        with open(cf, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("question", "").strip():
                    out.setdefault(int(r["id"]), r["question"])
    return out


def build_human(human_dir: Path, sql_dir: Path, name: str = "human") -> dict[str, list[dict]]:
    items = json.loads((human_dir / "sql_list.json").read_text(encoding="utf-8"))
    questions = _writer_questions(human_dir)
    recs = {r["sql"]: r for split in ("indist", "struct", "holdout")
            for r in json.loads(_sql_file(split, sql_dir).read_text(encoding="utf-8"))}
    pairs, bad = [], []
    for it in items:
        q = questions.get(int(it["id"]))
        if not q:
            continue
        rec = recs[it["sql"]]
        problems = question_gen.check_question(q.strip(), rec)
        if problems:
            bad.append((it["id"], q, problems))
            continue
        pairs.append(_pair(q, rec, id=it["id"], writer=it["writer"], split=it["split"]))
    by_writer = {w: sum(p["writer"] == w for p in pairs) for w in sorted({it["writer"] for it in items})}
    print(f"  [human] 질문 있는 문항 {len(pairs) + len(bad)} / {len(items)} (작성자별 {by_writer})")
    for i, q, problems in bad:
        print(f"  [human] 값 검사 실패로 제외: id {i} {q!r} -> {problems}")
    if not pairs:
        print("  [human] 쓸 문항이 없어 만들지 않음")
        return {}
    return {name: pairs}


def main() -> int:
    ap = argparse.ArgumentParser(description="2단계 평가셋 만들기")
    ap.add_argument("what", nargs="?", choices=["all", "composer", "ext", "human"], default="all")
    ap.add_argument("--sql-dir", default=str(PATHS.sql_dir), help=f"sql_gen 출력 (기본 {PATHS.sql_dir})")
    ap.add_argument("--eval-dir", default=str(PATHS.eval_dir), help=f"평가셋 폴더: pairs.json + sql.json (기본 {PATHS.eval_dir})")
    ap.add_argument("--seed", type=int, default=COMPOSER_SEED, help="composer: 조합기 문장 seed (학습 문장은 0)")
    ap.add_argument("--ext-rounds", nargs="*", default=None,
                    help=f"ext: 외부 말투 라운드 폴더들 (기본 {PATHS.raw_dir}/ext_*)")
    ap.add_argument("--human-dir", default=str(HUMAN_DIR), help=f"human: 사람 작성 세트 폴더 (기본 {HUMAN_DIR})")
    args = ap.parse_args()
    sql_dir, eval_dir = Path(args.sql_dir), Path(args.eval_dir)
    sets: dict[str, list[dict]] = {}
    drop: set[str] = set()
    if args.what in ("all", "composer"):
        sets |= build_composer(sql_dir, args.seed)
    if args.what in ("all", "ext"):
        rounds = [Path(r) for r in args.ext_rounds] if args.ext_rounds else sorted(PATHS.raw_dir.glob("ext_*"))
        ext = build_ext(rounds)
        sets |= ext
        if not args.ext_rounds:  # 라운드 전부를 다시 만들었으니 이번에 없는 ext_* 평가셋은 지운다
            pf = eval_dir / "pairs.json"
            old = {p[SET_KEY] for p in json.loads(pf.read_text(encoding="utf-8"))} if pf.is_file() else set()
            drop = {n for n in old if n.startswith("ext_") and n not in ext}
    if args.what in ("all", "human"):
        sets |= build_human(Path(args.human_dir), sql_dir)
    out = write_combined(sets, [_sql_file(s, sql_dir) for s in COMPOSER_SETS], eval_dir, drop)
    for name, pairs in sorted(sets.items()):
        print(f"[{name}] {len(pairs)}쌍")
    for name in sorted(drop):
        print(f"[{name}] 지움 (라운드 응답 없음)")
    print(f"-> {out} (+ sql.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
