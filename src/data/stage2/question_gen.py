# 역할: 2단계 외부 LLM 질문의 프롬프트 생성, 응답 검사와 학습 반영을 처리한다.
# 사용처: 2단계 학습 질문과 학습에 섞지 않을 외부 말투 평가 질문을 수집할 때 사용한다.

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
            # 과거 배포 묶음 응답 호환: 이 플랫폼이 응답한 배치의 SQL만 본다.
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
    ap.add_argument("command", choices=["make-prompts", "validate", "to-train"])
    ap.add_argument("--platform", default="claude", help="to-train: 옮길 플랫폼 응답")
    ap.add_argument("--out", required=True, help="라운드 폴더 (예: data_raw/stage2/eval_r1)")
    ap.add_argument("--split", choices=SPLITS, default="indist", help="make-prompts: SQL 을 뽑을 평가 분할")
    ap.add_argument("--sql", default=None, help=f"make-prompts: SQL 파일 (기본 {PATHS.sql_dir}/eval_<split>.json)")
    ap.add_argument("--n", type=int, default=120, help="make-prompts: SQL 수")
    ap.add_argument("--batch-size", type=int, default=20, help="make-prompts: 배치(LLM 세션) 하나의 SQL 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-dir", default="", help="make-prompts: 이전 라운드 폴더들 (쉼표 구분, 그 SQL 은 뽑지 않음)")
    args = ap.parse_args()
    excl = [d for d in args.exclude_dir.split(",") if d]
    if args.command == "make-prompts":
        make_prompts(args.out, args.split, args.n, args.batch_size, args.seed, args.sql, excl)
    elif args.command == "validate":
        validate(args.out)
    else:
        to_train(args.out, args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
