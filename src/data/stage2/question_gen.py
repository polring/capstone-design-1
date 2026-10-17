"""
2단계 LLM 질문 생성: 외부 말투 평가 질문 (계획서 3-4 "2단계 이후 평가 질문", 방식 A: SQL → 질문)과 학습 질문(--split train)

2단계 평가 분할(sql_gen 출력: eval_indist / eval_struct / eval_holdout)에서 SQL 을 뽑아, 외부 LLM 에 붙여넣을 배치 프롬프트를
만든다. SQL 하나마다 페르소나별 질문을 하나씩 받는다(칸 = 플랫폼 × 페르소나). 정답 SQL 은 이미 정해져 있고 LLM 은 질문만 쓴다.
같은 프롬프트를 여러 플랫폼에 돌려 플랫폼마다 응답 폴더를 따로 둔다.

프롬프트에는 스키마, 값 표기, SQL 의 뜻만 넣는다. 조합기 틀·표현 사전, 학습 문장, 라벨 규칙의 표현 목록은 넣지 않는다
(평가 문장이 학습 말투를 따라가지 않게). 배치 하나는 서로 독립된 LLM 세션 하나에 맡긴다(한 세션이 여러 배치를 맡으면 문장이
소수의 틀로 수렴한다). 검사 기준은 프롬프트에 쓰지 않는다.

사용법 (저장소 루트에서)
    python -m src.data.stage2.question_gen make-prompts --out data_raw/stage2/eval_r1 --split indist --n 120 --seed 0
        <out>/prompts/batch_XX.txt              LLM 에 붙여넣을 프롬프트 (배치 하나 = 새 세션 하나)
        <out>/round_sql.json                    id → SQL 레코드 (검증용)

    (사람이 batch_XX.txt 를 플랫폼마다 새 세션에 붙여넣고, 응답 JSON 을 <out>/responses/<플랫폼>/batch_XX.json 으로 저장)

    python -m src.data.stage2.question_gen validate --out data_raw/stage2/eval_r1
        <out>/pairs_<플랫폼>.json                검증 통과한 쌍 (question, sql = IN 값 질문 순서, sql_group = 정규형, persona …)
        <out>/validation_report.json            플랫폼 × 페르소나별 통과율 · 실패 사유 · 확인용 flag

    라운드는 data_raw/stage2/ 에서 검증하고, 직접 읽어 확인한 뒤 평가셋으로 쓴다. data/stage2/eval/<라운드>_<플랫폼>/ 은
    python -m src.data.stage2.eval_build ext 로 만든다. dev / 봉인 구분(계획서 3-4)은 라운드·플랫폼 단위로 정한다.

학습 질문: --split train 은 학습 SQL(data/stage2/sql/train.json)에서 뽑고, 평가와 다른 페르소나(TRAIN_PERSONAS)로 프롬프트를 만든다
(평가 질문과 같은 페르소나로 학습하면 외부 말투 평가가 분포 안 평가가 된다). 평가 SQL 로 만든 질문(ext_*)은 학습에 넣지 않는다
(holdout 리터럴은 학습 SQL 에 나오면 안 되고, 평가 SQL 이 학습에 들어가면 그 평가가 암기 점수가 된다).
    python -m src.data.stage2.question_gen make-prompts --out data_raw/stage2/llm_r1 --split train --n 200 --seed 0
    python -m src.data.stage2.question_gen validate --out data_raw/stage2/llm_r1
    python -m src.data.stage2.question_gen to-train --out data_raw/stage2/llm_r1 --platform claude
        → data/stage2/train/llm.json 에 넣는다 (쌍마다 round, source = llm:<플랫폼>:<페르소나>. 같은 라운드·플랫폼은 바꾼다.
          기존 학습 쌍과 질문이 같은데 SQL 이 다른 쌍은 뺀다). 옮긴 뒤 BPE 를 다시 만든다

배포 묶음 (1단계 새 말투 배포와 같은 방식): 다른 플랫폼(GPT, Gemini 등)을 쓰는 팀원에게 SQL 목록과 안내문(handout_template.md)만 준다.
프롬프트는 우리가 쓰지 않는다. 그 LLM 이 안내문의 BRIEF(규칙 포함)를 읽고 질문 생성 프롬프트를 직접 쓴 뒤 배치마다 새 세션에서
질문을 만든다. 페르소나도 그 LLM 이 정하므로 질문 자리는 q1~q5 로만 부른다. 학습 SQL 과 평가 SQL 을 섞어 한 묶음으로 줄 수 있다
(배포 id 는 묶음 안 번호, 받는 쪽은 어느 것이 평가용인지 모른다). 배치를 플랫폼별로 나눠 맡겨도 된다(validate 는 플랫폼마다 응답한 배치만 본다).
    python -m src.data.stage2.question_gen make-handout --out data_raw/stage2/llm_r3 --n 800 --batch-size 25 --seed 2 \
        --exclude-dir data_raw/stage2/llm_r1,data_raw/stage2/llm_r2 \
        --eval indist:80,struct:80,holdout:40 --eval-from data_raw/stage2/ext_r1 --eval-prefix data_raw/stage2/ext_r2
        <out>/handout/                          팀원에게 보낼 폴더 (README.md, sql/pilot.txt, sql/batch_XX.txt)
        <out>/round_sql.json, <out>/pilot/, data_raw/stage2/ext_r2_<분할>/   라운드별 검증용 메타, <out>/handout_map.json
    python -m src.data.stage2.question_gen collect-handout --out data_raw/stage2/llm_r3 --platform gpt --responses <돌려받은 폴더>
        응답을 라운드별 responses/<플랫폼>/ 으로 나누고 라운드마다 validate (플랫폼마다 한 번씩). 그 뒤 학습분은 to-train, 평가분은 eval_build ext
    프롬프트를 정해 주는 묶음: --prompt 로 이전 배포의 최종 프롬프트를 배치 파일마다 넣는다(파일럿 없음, --n 0 이면 평가 SQL 만).
    python -m src.data.stage2.question_gen make-handout --out data_raw/stage2/gpt_r3 --n 0 --batch-size 25 --seed 3 \
        --prompt <최종 프롬프트 파일> --eval indist:80,struct:80,holdout:40 --eval-from data_raw/stage2/ext_r2 --eval-prefix data_raw/stage2/ext_r3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.config import stage_paths
from src.data.stage1.question_gen import NUMBER_WORDS, literal_in_question
from src.data.stage2 import compose, schema
from src.data.stage2.clean_ambiguous import REASONS, drop_reason

PATHS = stage_paths(2)  # 2단계 SQL 로 만드는 질문이라 config.STAGE 와 무관하게 2단계 경로를 쓴다
SPLITS = ("indist", "struct", "holdout", "train")
ROUND_SQL = "round_sql.json"
ROUND_META = "round_meta.json"  # 분할, 페르소나 순서
REPORT = "validation_report.json"

# 페르소나 (계획서 3-4: 말투 차이는 페르소나로 만든다). 순서가 응답의 questions 순서다
PERSONAS = {
    "keyword": "types into a search box: a terse keyword fragment, not a full sentence",
    "chat": "casual chat: all lowercase, abbreviations, small typos in ordinary words (never in the values)",
    "nonnative": "non-native english speaker: simple words, slightly unusual grammar or word order, still clear",
    "verbose": "verbose and polite: one or two sentences that also say why they need it",
    "business": "a short work message to a colleague who has database access",
}
# 학습 질문용 페르소나. 평가 페르소나와 겹치지 않게 다른 말투를 고른다 (봉인 평가는 다른 플랫폼으로 따로 만든다)
TRAIN_PERSONAS = {
    "plain": "a plain, direct question in ordinary words",
    "spoken": "relaxed spoken style: contractions, filler words, may drop small words",
    "command": "an imperative instruction to a database assistant",
    "context": "a longer request that first describes the situation, then asks",
    "shorthand": "note-style shorthand: abbreviations, symbols, few function words",
}


def personas_for(split: str) -> dict[str, str]:
    return TRAIN_PERSONAS if split == "train" else PERSONAS

SCHEMA_BLURB = """\
domain: a shop's order-management database (one table per question, no joins).
- customers: customer_id, name (single lowercase word, e.g. marilyn), city, membership (bronze/silver/gold/platinum)
- items: item_id, item_name (single lowercase word, e.g. sofa), category, price (dollars), stock (units left in inventory)
- orders: order_id, customer_id, item_id, quantity (units in that order), status (pending/shipped/delivered/cancelled)
note: "stock" = inventory left; "quantity" = how many units were ordered. keep them distinct.\
"""


def rules_blurb(personas: dict[str, str] = PERSONAS) -> str:
    slots = "\n".join(f"   {i}. {name}: {desc}" for i, (name, desc) in enumerate(personas.items(), start=1))
    return f"""\
For EACH sql below, write {len(personas)} english questions, one per persona in this order:
{slots}

Every question must:
1. ask for exactly what the sql returns -- the same table, every condition, the sort order and its direction, and
   how many rows when there is a LIMIT. add nothing and leave nothing out. it must have only one sensible reading.
2. contain every value exactly as written in the sql (same spelling and digits; a price may be written 20, 20.00 or
   $20). write numbers as digits. for a LIKE pattern, say whether the name starts with, ends with, or contains the
   given letters. a list of values (IN) may be written in any order.
3. ask for what the sql selects: SELECT * means the full records; name / item_name / order_id means which ones or
   their names; customer_id / item_id on their own table means the id itself; any other column means that attribute.
   if the selected column is the same column the sql checks with =, the sql only confirms such rows exist -- ask it
   as an existence check.
4. not contain sql keywords or column names with underscores (say "item id", not item_id).

Return ONLY a JSON array, no commentary, no markdown code fences, in exactly this shape:
[
  {{"id": <id from input>, "questions": [{", ".join(f'"<{n}>"' for n in personas)}]}},
  ...
]
"""


def build_batch_prompt(batch: list[tuple[int, dict]], personas: dict[str, str] = PERSONAS) -> str:
    lines = [SCHEMA_BLURB, "", rules_blurb(personas), "sql list:"]
    lines += [f"- id={i} | {rec['sql']}" for i, rec in batch]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SQL 고르기
# ---------------------------------------------------------------------------

def shape_key(split: str, rec: dict) -> tuple:
    """고르게 뽑을 SQL 모양. struct 는 조합, holdout 은 미등장 모양 × 층, indist 는 테이블 × 연결 × IN × 정렬·LIMIT."""
    if split == "struct":
        return (rec["struct_combo"],)
    if split == "holdout":
        return (rec["holdout_shape"], rec["holdout_tier"])
    ops = {c["op"] for c in rec["conds"]}
    return (rec["table"], rec["conn"] or "-", "IN" in ops, "LIKE" in ops,
            "-" if rec["order"] is None else ("1" if rec["limit"] == 1 else "n" if rec["limit"] else "sort"))


def pick(rows: list[dict], split: str, n: int, seed: int, exclude: set[str]) -> list[int]:
    """모양마다 돌아가며 하나씩 뽑는다 (드문 모양도 들어가게). 반환은 rows 안의 인덱스."""
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["sql"] not in exclude:
            groups[shape_key(split, r)].append(i)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    out: list[int] = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return sorted(out)


def make_prompts(out_dir: str, split: str, n: int, batch_size: int, seed: int, sql_path: str | None = None,
                 exclude_dirs: list[str] | None = None) -> None:
    sql_path = sql_path or str(PATHS.sql_dir / ("train.json" if split == "train" else f"eval_{split}.json"))
    personas = personas_for(split)
    with open(sql_path, encoding="utf-8") as f:
        rows = json.load(f)
    exclude: set[str] = set()
    for d in exclude_dirs or []:
        with open(os.path.join(d, ROUND_SQL), encoding="utf-8") as f:
            exclude |= {r["sql"] for r in json.load(f).values()}
    ids = pick(rows, split, n, seed, exclude)
    rng = random.Random(seed + 1)
    rng.shuffle(ids)

    os.makedirs(os.path.join(out_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    for bi, b in enumerate(batches, start=1):
        with open(os.path.join(out_dir, "prompts", f"batch_{bi:02d}.txt"), "w", encoding="utf-8") as f:
            f.write(build_batch_prompt([(i, rows[i]) for i in b], personas))
    meta = {i: dict(rows[i], split=split, batch=bi) for bi, b in enumerate(batches, start=1) for i in b}
    with open(os.path.join(out_dir, ROUND_SQL), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, ROUND_META), "w", encoding="utf-8") as f:
        json.dump({"split": split, "personas": list(personas), "sql": sql_path, "seed": seed}, f, ensure_ascii=False, indent=1)
    print(f"[{split}] SQL {len(ids)}개 ({len({shape_key(split, rows[i]) for i in ids})}개 모양) -> 배치 {len(batches)}개 "
          f"(배치 크기 {batch_size}, SQL 당 질문 {len(personas)}개)")
    print(f"프롬프트: {out_dir}/prompts/batch_01.txt ~ batch_{len(batches):02d}.txt (배치마다 새 세션)")
    print(f"응답 저장: {out_dir}/responses/<플랫폼>/batch_XX.json")


HANDOUT_DIR = PATHS.data_dir / "handout"  # 배포 묶음 안내문 틀
HANDOUT_TEMPLATE = HANDOUT_DIR / "handout_template.md"
HANDOUT_FIXED_TEMPLATE = HANDOUT_DIR / "handout_fixed_template.md"  # 프롬프트를 배치 파일에 넣어 주는 묶음
HANDOUT_SLOTS = [f"q{i}" for i in range(1, 6)]  # BRIEF 가 SQL 당 질문 5개를 정한다. 말투는 받는 LLM 이 정한다
HANDOUT_MAP = "handout_map.json"  # 묶음 id → (라운드 폴더, 원래 id, 배치 파일)


def _load_exclude(dirs: list[str] | None) -> set[str]:
    out: set[str] = set()
    for d in dirs or []:
        with open(os.path.join(d, ROUND_SQL), encoding="utf-8") as f:
            out |= {r["sql"] for r in json.load(f).values()}
    return out


def _write_round(d: Path, items: dict[int, dict], split: str, source: str, seed: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / ROUND_SQL).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / ROUND_META).write_text(json.dumps({"split": split, "personas": HANDOUT_SLOTS, "sql": source, "seed": seed,
                                            "handout": True}, ensure_ascii=False, indent=1), encoding="utf-8")


def make_handout(out_dir: str, n: int, pilot_n: int, batch_size: int, seed: int, sql_path: str | None = None,
                 exclude_dirs: list[str] | None = None, eval_counts: dict[str, int] | None = None,
                 eval_from: str | None = None, eval_prefix: str | None = None,
                 prompt: str | None = None) -> None:
    """배포 묶음: 학습 SQL n개 + 평가 SQL(분할별 eval_counts)을 섞어 배치로 나누고, 파일럿은 학습 SQL 로 따로 둔다.
    - 학습 SQL 은 train.json 에서 모양별로 돌아가며 뽑는다(exclude_dirs 라운드의 SQL 제외). 라운드 폴더 = out_dir.
    - 평가 SQL 은 eval_from 이 있으면 <eval_from>_<분할>/round_sql.json(이전 외부 말투 라운드)에서 뽑는다. 1단계 새 말투 배포처럼
      같은 SQL 에서 다른 플랫폼의 질문과 바로 비교하려는 것이다. 없으면 eval_<분할>.json 에서. 라운드 폴더 = <eval_prefix>_<분할>.
    - 배포 파일의 id 는 묶음 안 번호(1부터)다. SQL 파일마다 인덱스가 겹쳐서다. 원래 (라운드, id) 는 handout_map.json 에 두고
      collect-handout 이 응답을 라운드별로 되돌린다. 받는 쪽은 어느 SQL 이 평가용인지 모른다.
    - prompt(이전 배포에서 받은 최종 프롬프트 파일)를 주면 배치 파일마다 프롬프트 + SQL 목록을 통째로 넣는다. 받는 쪽은 프롬프트를
      쓰지 않으므로 파일럿을 만들지 않고, 안내문은 handout_fixed_template.md 다. n = 0 이면 평가 SQL 만 담는다(학습 라운드 없음)."""
    out = Path(out_dir)
    eval_counts = eval_counts or {}
    prompt_text = Path(prompt).read_text(encoding="utf-8-sig").strip() if prompt else None
    if prompt_text:
        pilot_n = 0
    rounds = ({out: "train"} if n else {}) | {Path(f"{eval_prefix}_{s}"): s for s in eval_counts}
    if eval_counts and not eval_prefix:
        raise SystemExit("평가 SQL 을 넣으려면 --eval-prefix 가 필요하다")
    if taken := [str(d) for d in rounds if (d / ROUND_SQL).exists()]:
        raise SystemExit(f"이미 있는 라운드 폴더: {taken} (지우고 다시 만든다)")

    sql_path = sql_path or str(PATHS.sql_dir / "train.json")
    with open(sql_path, encoding="utf-8") as f:
        rows = json.load(f)
    exclude = _load_exclude(exclude_dirs)
    train_ids = pick(rows, "train", n, seed, exclude)
    pilot_ids = pick(rows, "train", pilot_n, seed + 1, exclude | {rows[i]["sql"] for i in train_ids})
    main = [(out, i, rows[i]) for i in train_ids]
    sources = {out: sql_path}
    for split, cnt in eval_counts.items():
        if eval_from:
            src = Path(f"{eval_from}_{split}") / ROUND_SQL
            cand = [(int(k), v) for k, v in json.loads(src.read_text(encoding="utf-8")).items()]
        else:
            src = PATHS.sql_dir / f"eval_{split}.json"
            cand = list(enumerate(json.loads(src.read_text(encoding="utf-8"))))
        rd = Path(f"{eval_prefix}_{split}")
        sources[rd] = str(src)
        main += [(rd, cand[j][0], cand[j][1]) for j in pick([r for _, r in cand], split, cnt, seed, set())]
    rng = random.Random(seed + 2)
    rng.shuffle(main)
    rng.shuffle(pilot_ids)

    (out / "handout" / "sql").mkdir(parents=True, exist_ok=True)
    hmap: dict[int, dict] = {}
    per_round: dict[Path, dict[int, dict]] = {d: {} for d in rounds}
    batches = [main[i:i + batch_size] for i in range(0, len(main), batch_size)]
    hid = 0
    for bi, b in enumerate(batches, start=1):
        lines = []
        for rd, oid, rec in b:
            hid += 1
            lines.append(f"id={hid} | {rec['sql']}\n")
            hmap[hid] = {"round": str(rd), "id": oid, "file": f"batch_{bi:02d}"}
            per_round[rd][oid] = {k: v for k, v in rec.items() if k != "batch"} | {"split": rounds[rd], "batch": bi}
        text = "".join(lines)
        if prompt_text:
            text = f"{prompt_text}\n\n{text}"
        (out / "handout" / "sql" / f"batch_{bi:02d}.txt").write_text(text, encoding="utf-8")
    pilot_lines, pilot_meta = [], {}
    for oid in pilot_ids:
        hid += 1
        pilot_lines.append(f"id={hid} | {rows[oid]['sql']}\n")
        hmap[hid] = {"round": str(out / "pilot"), "id": oid, "file": "pilot"}
        pilot_meta[oid] = dict(rows[oid], split="train", batch=0)
    if pilot_ids:
        (out / "handout" / "sql" / "pilot.txt").write_text("".join(pilot_lines), encoding="utf-8")

    for rd, items in per_round.items():
        _write_round(rd, items, rounds[rd], sources[rd], seed)
    if pilot_ids:
        _write_round(out / "pilot", pilot_meta, "train", sql_path, seed)
    (out / HANDOUT_MAP).write_text(json.dumps(hmap, ensure_ascii=False, indent=1), encoding="utf-8")

    doc = (HANDOUT_FIXED_TEMPLATE if prompt_text else HANDOUT_TEMPLATE).read_text(encoding="utf-8")
    for key, val in {"N_SQL": len(main), "PILOT_SIZE": len(pilot_ids), "BATCH_SIZE": batch_size,
                     "LAST_BATCH": f"{len(batches):02d}"}.items():
        doc = doc.replace(f"<<{key}>>", str(val))
    (out / "handout" / "README.md").write_text(doc, encoding="utf-8")
    print(f"SQL {len(main)}개 -> 배치 {len(batches)}개 (배치 크기 {batch_size}), 파일럿 {len(pilot_ids)}개")
    for rd, items in per_round.items():
        print(f"  {rd}: {rounds[rd]} {len(items)}개 ({len({shape_key(rounds[rd], r) for r in items.values()})}개 모양)")
    print(f"배포: {out / 'handout'} (이 폴더만 보낸다)")


def collect_handout(out_dir: str, platform: str, returned: str) -> None:
    """돌려받은 폴더(<returned>/responses/batch_XX.json, <returned>/pilot_vN.json)의 응답을 묶음 id → 원래 id 로 바꿔
    라운드마다 <라운드>/responses/<platform>/batch_XX.json 으로 나누고, 라운드마다 validate 한다.
    파일럿은 프롬프트 버전마다 <out>/pilot/responses/<platform>_vN/ 으로 둔다(validate 가 버전을 플랫폼처럼 나란히 보인다).
    응답은 고치지 않는다: 감싼 코드 블록만 벗기고, 깨진 JSON 파일은 알리고 건너뛴다(그 배치는 validate 에 안 보이니 출력의 배치 목록을 확인한다)."""
    out, ret = Path(out_dir), Path(returned)
    hmap = {int(k): v for k, v in json.loads((out / HANDOUT_MAP).read_text(encoding="utf-8")).items()}
    files = [(f, platform) for f in sorted((ret / "responses").glob("*.json"))]
    files += [(f, f"{platform}_{f.stem.removeprefix('pilot_')}") for f in sorted(ret.glob("pilot_v*.json"))]
    grouped: dict[tuple[str, str, str], list] = defaultdict(list)
    for f, plat in files:
        text = f.read_text(encoding="utf-8-sig").strip()
        if fence := re.fullmatch(r"```[a-zA-Z]*\s*\n(.*)\n\s*```", text, re.S):
            text = fence.group(1)  # 코드 블록으로 감싸 온 응답 (내용은 그대로)
        try:
            batch = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[건너뜀] {f.name}: JSON 파싱 실패 ({e})")
            continue
        for item in batch if isinstance(batch, list) else []:
            try:
                m = hmap[int(item["id"])]
            except (KeyError, TypeError, ValueError):
                print(f"[건너뜀] {f.name}: 모르는 id {item.get('id') if isinstance(item, dict) else item!r}")
                continue
            grouped[(m["round"], plat, m["file"])].append(dict(item, id=m["id"]))
    for (rd, plat, name), items in sorted(grouped.items()):
        d = Path(rd) / "responses" / plat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    for rd in sorted({rd for rd, _, _ in grouped}):
        print(f"--- {rd}")
        validate(rd)


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def _coltype(table: str, col: str) -> str:
    if schema.is_text(table, col):
        return "text"
    return "float" if (table, col) in schema.FLOAT_COLUMNS else "int"


# 숫자 없이 말해도 되는 고정값 조건 (L16: out of stock = stock = 0, in stock = stock > 0). 조합기 값 전용 키와 같다
VALUE_KEYS = compose.value_keys(json.loads(compose.LEXICON_PATH.read_text(encoding="utf-8")))


def literal_problems(q: str, rec: dict) -> list[str]:
    """SQL 의 값(=·!=·IN·비교·BETWEEN 리터럴, LIKE 조각, LIMIT 2 이상)이 질문에 그대로 있는지. 고정값 조건은 보지 않는다."""
    out = []
    t = rec["table"]
    for c in rec["conds"]:
        if c["op"] != "IN" and f"{t}.{c['col']} {c['op']} {c['val']}" in VALUE_KEYS:
            continue
        if c["op"] == "LIKE":
            frag = c["val"].strip("%")
            if not re.search(rf"(?<![a-z]){re.escape(frag)}(?![a-z])", q):
                out.append(f"LIKE 조각 {frag!r} 없음")
            continue
        vals = list(c["val"]) if c["op"] == "IN" else [c["val"]] + ([c["val2"]] if c["op"] == "BETWEEN" else [])
        for v in vals:
            if not literal_in_question(q, v, _coltype(t, c["col"])):
                out.append(f"값 {v!r} 없음")
    if rec["limit"] and rec["limit"] > 1 and not literal_in_question(q, rec["limit"], "int"):
        out.append(f"LIMIT {rec['limit']} 없음")
    return out


def check_question(q: str, rec: dict) -> list[str]:
    """하드 실패 사유 (비면 통과). 라벨(IN 값 질문 순서)을 못 정하는 문장도 실패다.
    추론은 질문을 소문자로 바꿔 넣으므로(generate.py) 값 검사도 소문자로 한다. SQL 키워드 검사만 원문으로 한다."""
    raw, q = q, q.lower()
    fails = literal_problems(q, rec)
    if snake := re.findall(r"\b[a-z]+_[a-z_]+\b", q):
        fails.append(f"밑줄 컬럼 이름 {snake}")
    if re.search(r"\b(SELECT|WHERE|FROM|LIMIT|ORDER BY)\b", raw):
        fails.append("SQL 키워드")
    if not fails:
        try:
            compose.question_order_sql(rec, q)
        except compose.ComposeError:
            fails.append("IN 값 위치를 정할 수 없음")
    return fails


def soft_flags(q: str, rec: dict) -> list[str]:
    """실패는 아니지만 읽어 볼 것: 숫자 단어("these two products" — 값의 숫자 표기는 literal_problems 가 본다),
    stock/quantity 혼동, 존재 확인 SQL 인데 확인 질문이 아님."""
    flags = []
    q = q.lower()
    if words := sorted(set(re.findall(r"[a-z]+", q)) & NUMBER_WORDS):
        flags.append(f"숫자 단어 {words}")
    cols = {c["col"] for c in rec["conds"]} | {rec["select"]} | ({rec["order"]["col"]} if rec["order"] else set())
    if "stock" in cols and "quantity" not in cols and re.search(r"\border(ed|s)?\b", q) and rec["table"] == "items":
        flags.append("items 의 stock 인데 order 표현")
    if "quantity" in cols and "stock" in q:
        flags.append("quantity 인데 stock 표현")
    if compose.role_of(rec) == "exists" and not re.search(r"\b(is there|are there|exists?|any|anyone|confirm|verify|check|do we)\b", q):
        flags.append("존재 확인 SQL 인데 확인 질문이 아님")
    return flags


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


def load_responses(resp_dir: Path) -> tuple[dict[int, list], list[str]]:
    out: dict[int, list] = {}
    problems: list[str] = []
    for f in sorted(resp_dir.glob("*.json")):
        try:
            batch = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"{f.name}: JSON 파싱 실패 ({e})")
            continue
        if not isinstance(batch, list):
            problems.append(f"{f.name}: JSON 배열이 아님")
            continue
        for item in batch:
            try:
                qid, qs = int(item["id"]), item["questions"]
            except (KeyError, TypeError, ValueError) as e:
                problems.append(f"{f.name}: 항목 형식 오류 ({e})")
                continue
            if qid in out:
                problems.append(f"{f.name}: id {qid} 중복")
            out[qid] = qs
    return out, problems


def validate(out_dir: str) -> dict:
    out = Path(out_dir)
    meta = {int(k): v for k, v in json.loads((out / ROUND_SQL).read_text(encoding="utf-8")).items()}
    rmeta = json.loads((out / ROUND_META).read_text(encoding="utf-8")) if (out / ROUND_META).exists() else {}
    personas = rmeta.get("personas", list(PERSONAS))
    platforms = sorted(p.name for p in (out / "responses").iterdir() if p.is_dir()) if (out / "responses").is_dir() else []
    report: dict = {"n_sql": len(meta), "personas": personas, "platforms": {}}
    for plat in platforms:
        responses, problems = load_responses(out / "responses" / plat)
        pairs, failures, flags = [], [], []
        seen, passed = Counter(), Counter()
        rule_dropped: dict[str, Counter] = defaultdict(Counter)
        plat_meta = meta
        if rmeta.get("handout"):
            # 배포 묶음은 플랫폼마다 배치를 나눠 맡을 수 있다: 이 플랫폼이 응답한 배치의 SQL 만 본다 (배치 파일이 통째로 빠지면 여기서 안 보인다)
            got = {0 if f.stem.startswith("pilot") else int(f.stem.removeprefix("batch_"))
                   for f in (out / "responses" / plat).glob("*.json")}
            plat_meta = {k: v for k, v in meta.items() if v["batch"] in got}
            print(f"[{plat}] 응답 배치 {len(got)}개: {sorted(got)}")
        for qid, rec in sorted(plat_meta.items()):
            qs = responses.get(qid)
            if qs is None:
                failures.append({"id": qid, "sql": rec["sql"], "reasons": ["응답 없음"]})
                continue
            if len(qs) != len(personas):
                failures.append({"id": qid, "sql": rec["sql"], "reasons": [f"질문 {len(qs)}개 (기대 {len(personas)}개)"]})
            dup = set()
            for persona, q in zip(personas, qs):
                seen[persona] += 1
                if not isinstance(q, str) or not q.strip():
                    failures.append({"id": qid, "persona": persona, "sql": rec["sql"], "reasons": ["빈 질문"]})
                    continue
                q = q.strip()
                fails = check_question(q, rec)
                if _norm(q) in dup:
                    fails.append("같은 SQL 안 중복")
                dup.add(_norm(q))
                if fails:
                    failures.append({"id": qid, "persona": persona, "sql": rec["sql"], "question": q, "reasons": fails})
                    continue
                # 라벨 규칙 판정: 질문이 라벨 규칙상 다른 SQL 을 가리키면 뺀다 (계획서 3-4 "검증", 출처별로 버린 수를 기록)
                if rule := drop_reason({"question": q, "sql": rec["sql"]}):
                    rule_dropped[persona][rule] += 1
                    failures.append({"id": qid, "persona": persona, "sql": rec["sql"], "question": q,
                                     "reasons": [f"규칙 불일치 {REASONS[rule]}"]})
                    continue
                passed[persona] += 1
                pair = {"question": q, "sql": compose.question_order_sql(rec, q.lower()), "sql_group": rec["sql"],
                        "persona": persona, "platform": plat, "split": rec["split"], "id": qid}
                for k in ("struct_combo", "holdout_shape", "holdout_tier"):
                    if k in rec:
                        pair[k] = rec[k]
                pairs.append(pair)
                if sf := soft_flags(q, rec):
                    flags.append({"id": qid, "persona": persona, "question": q, "sql": rec["sql"], "flags": sf})
        # 같은 질문이 서로 다른 SQL 에 붙으면 라벨 충돌: 양쪽 다 뺀다
        by_q = defaultdict(set)
        for p in pairs:
            by_q[_norm(p["question"])].add(p["sql_group"])
        clash = {q for q, s in by_q.items() if len(s) > 1}
        if clash:
            for p in pairs:
                if _norm(p["question"]) in clash:
                    passed[p["persona"]] -= 1
                    failures.append({"id": p["id"], "persona": p["persona"], "sql": p["sql_group"],
                                     "question": p["question"], "reasons": ["다른 SQL 과 같은 질문"]})
            pairs = [p for p in pairs if _norm(p["question"]) not in clash]
        (out / f"pairs_{plat}.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
        reasons = Counter(r.split(" ")[0] if r.startswith(("값", "LIKE", "LIMIT")) else r
                          for f in failures for r in f["reasons"])
        report["platforms"][plat] = {
            "n_sql": len(plat_meta),
            "n_pairs": len(pairs),
            "pass_rate": {p: round(passed[p] / seen[p], 4) if seen[p] else None for p in personas},
            "rule_dropped": {p: dict(rule_dropped[p]) for p in personas},
            "failure_reasons": dict(reasons.most_common()),
            "response_problems": problems,
            "n_soft_flags": len(flags),
            "failures": failures,
            "soft_flags": flags,
        }
        print(f"[{plat}] 통과 {len(pairs)}쌍 / 페르소나별 통과율 "
              + ", ".join(f"{p} {passed[p]}/{seen[p]}" for p in personas)
              + f" | 규칙 불일치 {sum(sum(c.values()) for c in rule_dropped.values())}"
              + f" | 확인용 flag {len(flags)}" + (f" | 응답 문제 {len(problems)}" if problems else ""))
    (out / REPORT).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if not platforms:
        print(f"응답 폴더 없음: {out}/responses/<플랫폼>/")
    return report


LLM_TRAIN_FILE = "llm.json"  # LLM 학습 질문을 모두 담는 학습 파일 (쌍마다 round · source = llm:<플랫폼>:<페르소나>)


def to_train(out_dir: str, platform: str, train_dir: Path = PATHS.train_dir) -> int:
    """검증 통과 쌍(pairs_<플랫폼>.json)을 학습 폴더의 llm.json 에 넣는다. 같은 (라운드, 플랫폼) 쌍이 이미 있으면 바꾼다.
    학습 SQL 로 만든 라운드만 받는다. 질문은 소문자로 바꾼다(다른 학습 쌍과 추론 입력이 소문자이고 토크나이저는 대소문자를 구분한다).
    다른 학습 쌍과 질문이 같은데 SQL(sql_group)이 다른 쌍은 라벨 충돌이라 뺀다."""
    out = Path(out_dir)
    meta = json.loads((out / ROUND_META).read_text(encoding="utf-8"))
    if meta["split"] != "train":
        raise SystemExit(f"평가 SQL 라운드({meta['split']})는 학습에 넣지 않는다")
    pairs = json.loads((out / f"pairs_{platform}.json").read_text(encoding="utf-8"))
    rnd, source = out.name, f"llm:{platform}:"
    target = train_dir / LLM_TRAIN_FILE
    mine = lambda p: p.get("round") == rnd and p["source"].startswith(source)
    others = [p for p in json.loads(target.read_text(encoding="utf-8")) if not mine(p)] if target.exists() else []
    seen: dict[str, str] = {_norm(p["question"]): p.get("sql_group", p["sql"]) for p in others}
    for f in sorted(train_dir.glob("*.json")):
        if f != target:
            seen.update({_norm(p["question"]): p.get("sql_group", p["sql"]) for p in json.loads(f.read_text(encoding="utf-8"))})
    keep = [{"question": p["question"].strip().lower(), "sql": p["sql"], "sql_group": p["sql_group"],
             "source": f"{source}{p['persona']}", "round": rnd}
            for p in pairs if seen.get(_norm(p["question"]), p["sql_group"]) == p["sql_group"]]
    target.write_text(json.dumps(others + keep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{rnd} {platform} {len(keep)}쌍 -> {target} (모두 {len(others) + len(keep)}쌍, 기존 학습 쌍과 라벨 충돌 "
          f"{len(pairs) - len(keep)}개 제외). BPE 를 다시 만든다")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="2단계 외부 말투 평가 질문 생성 (방식 A: SQL → 질문)")
    ap.add_argument("command", choices=["make-prompts", "make-handout", "collect-handout", "validate", "to-train"])
    ap.add_argument("--platform", default="claude", help="to-train: 옮길 플랫폼 응답 / collect-handout: 응답을 넣을 플랫폼 이름")
    ap.add_argument("--out", required=True, help="라운드 폴더 (예: data_raw/stage2/eval_r1)")
    ap.add_argument("--split", choices=SPLITS, default="indist", help="make-prompts: SQL 을 뽑을 평가 분할")
    ap.add_argument("--sql", default=None, help=f"make-prompts: SQL 파일 (기본 {PATHS.sql_dir}/eval_<split>.json)")
    ap.add_argument("--n", type=int, default=120, help="make-prompts: SQL 수")
    ap.add_argument("--batch-size", type=int, default=20, help="make-prompts: 배치(LLM 세션) 하나의 SQL 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-dir", default="", help="make-prompts / make-handout: 이전 라운드 폴더들 (쉼표 구분, 그 SQL 은 뽑지 않음)")
    ap.add_argument("--pilot-n", type=int, default=25, help="make-handout: 파일럿 SQL 수")
    ap.add_argument("--eval", default="", help="make-handout: 섞어 넣을 평가 SQL 수 (예: indist:80,struct:80,holdout:40)")
    ap.add_argument("--eval-from", default=None, help="make-handout: 평가 SQL 을 뽑을 이전 라운드 접두 (예: data_raw/stage2/ext_r1 → ext_r1_<분할>)")
    ap.add_argument("--eval-prefix", default=None, help="make-handout: 평가 라운드 폴더 접두 (예: data_raw/stage2/ext_r2 → ext_r2_<분할>)")
    ap.add_argument("--prompt", default=None, help="make-handout: 배치 파일마다 넣을 프롬프트 파일 (주면 파일럿 없이 배치만 돌리는 묶음)")
    ap.add_argument("--responses", default=None, help="collect-handout: 팀원이 돌려준 폴더 (responses/, pilot_vN.json)")
    args = ap.parse_args()
    excl = [d for d in args.exclude_dir.split(",") if d]
    if args.command == "make-prompts":
        make_prompts(args.out, args.split, args.n, args.batch_size, args.seed, args.sql, excl)
    elif args.command == "make-handout":
        counts = {s: int(c) for s, c in (x.split(":") for x in args.eval.split(",") if x)}
        if bad := set(counts) - {"indist", "struct", "holdout"}:
            raise SystemExit(f"--eval 분할은 indist / struct / holdout 만: {sorted(bad)}")
        make_handout(args.out, args.n, args.pilot_n, args.batch_size, args.seed, args.sql, excl,
                     counts, args.eval_from, args.eval_prefix, args.prompt)
    elif args.command == "collect-handout":
        if not args.responses:
            raise SystemExit("--responses 가 필요하다")
        collect_handout(args.out, args.platform, args.responses)
    elif args.command == "validate":
        validate(args.out)
    else:
        to_train(args.out, args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
