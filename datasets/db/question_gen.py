"""
Text-to-SQL 1단계 LLM 질문 생성 파이프라인 (v1, 수동 호출 모드)

sql_gen.py 가 만든 sql_train.json 에서 SQL을 뽑아, 외부 LLM에게 보낼 배치 프롬프트를
텍스트 파일로 만든다. API를 직접 호출하지 않고, 사람이 프롬프트를 챗봇에 붙여넣고
받아온 응답을 다시 이 스크립트로 검증하는 흐름이다.

SQL 정답은 이미 sql_gen.py 가 정했다. LLM은 그 SQL에 맞는 자연어 질문만 만들고,
질문이 규칙(리터럴 포함, 소문자, 숫자 단어 금지 등)을 지키는지는 항상 코드가 검사한다
(CLAUDE.md 스크래치 원칙: LLM은 SQL 정답을 만들거나 수정하지 않는다).

사용법
    python question_gen.py make-prompts --out data/pilot --n 200 --batch-size 25 --seed 0
        data/pilot/prompts/batch_XX.txt          LLM에 붙여넣을 프롬프트
        data/pilot/pilot_sql.json                배치별 SQL과 메타데이터 (검증용)

    (사람이 각 batch_XX.txt 를 LLM에 붙여넣고, 응답 JSON을
     data/pilot/responses/batch_XX.json 으로 저장)

    python question_gen.py validate --out data/pilot
        data/pilot/pilot_train_pairs.json        검증 통과한 (질문, SQL) 쌍
        data/pilot/validation_report.json        통과율 · 실패 사유 · 수동 확인용 flag
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

# ---------------------------------------------------------------------------
# 1. SQL 샘플링 (sql_train.json 의 (table, where_col) 분포를 유지한 채 n개 축소)
# ---------------------------------------------------------------------------

def stratified_sample(entries: list[dict], n_total: int, seed: int) -> list[tuple[int, dict]]:
    """(원본 인덱스, entry) 쌍을 (table, where_col) 조합 비율을 유지해 n_total개 추출."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(entries):
        groups.setdefault((e["table"], e["where_col"]), []).append(i)

    total = len(entries)
    quotas = {k: round(len(v) * n_total / total) for k, v in groups.items()}
    diff = n_total - sum(quotas.values())
    keys = sorted(quotas)  # 결정적 순서로 나머지 보정
    i = 0
    while diff != 0:
        k = keys[i % len(keys)]
        if diff > 0:
            quotas[k] += 1
            diff -= 1
        elif quotas[k] > 0:
            quotas[k] -= 1
            diff += 1
        i += 1

    picked: list[tuple[int, dict]] = []
    for k, idxs in groups.items():
        chosen = rng.sample(idxs, min(quotas[k], len(idxs)))
        picked.extend((i, entries[i]) for i in chosen)
    rng.shuffle(picked)
    return picked


# ---------------------------------------------------------------------------
# 2. 프롬프트 생성
# ---------------------------------------------------------------------------

SCHEMA_BLURB = """\
domain: a shop's order-management database (customers, items, orders).
- customers: customer_id, name (single word, e.g. marilyn), city, membership (bronze/silver/gold/platinum)
- items: item_id, item_name (single word, e.g. sofa), category, price, stock (units left in inventory)
- orders: order_id, customer_id, item_id, quantity (units in that order), status (pending/shipped/delivered/cancelled)
note: "stock" = inventory left; "quantity" = how many units were ordered. These are easy to confuse -- keep them distinct.\
"""

RULES_BLURB = """\
For EACH sql below, write 5 different english questions that a person could ask to get exactly that sql as the answer.
Rules (all must hold for every question):
1. all lowercase, no punctuation-heavy stylization.
2. the question MUST literally contain the where-value exactly as given (same spelling/digits). do not paraphrase it.
   e.g. if the value is "marilyn", the word "marilyn" must appear; if the value is 1864, the digits "1864" must appear.
3. do NOT replace item/category words with a synonym or a broader word (e.g. never turn "laptop" into "computer",
   "sneakers" into "shoes", or "stock = 0" into "sold out").
4. do NOT spell numbers out as words ("twenty"); always use digits.
5. do NOT write a question that needs another table's info (no joins). only use facts available from THIS one sql.
6. make the 5 questions genuinely different in register/phrasing (e.g. casual, formal, terse keyword-style,
   a direct command, a "how many/what/which" question) -- not just synonyms of each other.
7. if select is "stock", ask about inventory/units left, not about how many were ordered. if select is "quantity",
   ask about the order amount, not inventory. do not mix these two up.
8. if select is "*", ask a general "tell me about / show me" style question, not one naming a specific column.
9. never write a raw sql column name with an underscore (item_id, order_id, customer_id, item_name) in the
   question text. use natural words instead -- "item id" (with a space) or just "item"/"order"/"customer",
   never the literal snake_case identifier.

Return ONLY a JSON array, no commentary, no markdown code fences, in exactly this shape:
[
  {"id": <id from input>, "questions": ["q1", "q2", "q3", "q4", "q5"]},
  ...
]
"""


def format_literal_for_prompt(where_val, coltype: str) -> str:
    if coltype == "float":
        return f"{where_val:.2f}"
    return str(where_val)


def build_batch_prompt(batch: list[dict]) -> str:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import db_gen  # COLUMN_TYPES 대신 여기서 직접 판단하지 않고 sql_gen 재사용
    import sql_gen

    lines = [SCHEMA_BLURB, "", RULES_BLURB, "", "sql list:"]
    for item in batch:
        e = item["entry"]
        coltype = sql_gen.COLUMN_TYPES[(e["table"], e["where_col"])]
        lit = format_literal_for_prompt(e["where_val"], coltype)
        lines.append(
            f'- id={item["id"]} | sql: {e["sql"]} | select={e["select"]} | '
            f'where_col={e["where_col"]} | where_value={lit} | exists={not e["is_nonexistent"]}'
        )
    return "\n".join(lines)


def make_prompts(out_dir: str, train_path: str, n: int, batch_size: int, seed: int,
                  exclude_dirs: list[str] | None = None) -> None:
    with open(train_path, encoding="utf-8") as f:
        entries = json.load(f)

    if exclude_dirs:
        already_used: set[str] = set()
        for d in exclude_dirs:
            with open(os.path.join(d, "pilot_sql.json"), encoding="utf-8") as f:
                already_used |= {v["sql"] for v in json.load(f).values()}
        before = len(entries)
        entries = [e for e in entries if e["sql"] not in already_used]
        print(f"이전 {len(exclude_dirs)}개 라운드에서 쓴 SQL {before - len(entries)}개를 후보에서 제외 "
              f"(남은 후보 {len(entries)}개)")

    picked = stratified_sample(entries, n, seed)
    items = [{"id": idx, "entry": e} for idx, e in picked]

    os.makedirs(os.path.join(out_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "responses"), exist_ok=True)

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for bi, batch in enumerate(batches, start=1):
        prompt = build_batch_prompt(batch)
        path = os.path.join(out_dir, "prompts", f"batch_{bi:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt)

    pilot_sql = {item["id"]: item["entry"] for item in items}
    with open(os.path.join(out_dir, "pilot_sql.json"), "w", encoding="utf-8") as f:
        json.dump(pilot_sql, f, ensure_ascii=False, indent=2)

    print(f"SQL {len(items)}개 -> {len(batches)}개 배치")
    print(f"프롬프트: {os.path.join(out_dir, 'prompts')}/batch_01.txt ~ batch_{len(batches):02d}.txt")
    print(f"응답을 저장할 위치: {os.path.join(out_dir, 'responses')}/batch_01.json ~ batch_{len(batches):02d}.json")
    print(f"메타데이터: {os.path.join(out_dir, 'pilot_sql.json')}")


# ---------------------------------------------------------------------------
# 3. 자동 검증 (계획서 4-3/4-4 규칙 + 3-3 완료조건)
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    # "one"은 "is X one of our customers" 같은 일상 표현에 흔히 쓰여 오탐이 많아 제외.
    # 숫자 "1"을 진짜 단어로 쓴 경우는 별도 리터럴 포함 검사가 이미 걸러낸다.
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
}

KNOWN_SYNONYM_SWAPS = {
    "computer": "laptop", "shoes": "sneakers", "sold out": "stock = 0", "sofa": "couch",
}


def literal_in_question(q: str, where_val, coltype: str) -> bool:
    if coltype == "text":
        return str(where_val).lower() in q
    if coltype == "float":
        return f"{where_val:.2f}" in q or f"{where_val:.2f}".rstrip("0").rstrip(".") in q
    return str(where_val) in q


def check_question(q: str, entry: dict) -> list[str]:
    """하나의 질문에 대한 하드 실패 이유 목록. 비어있으면 통과."""
    fails = []
    if q != q.lower():
        fails.append("소문자 규칙 위반")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sql_gen
    coltype = sql_gen.COLUMN_TYPES[(entry["table"], entry["where_col"])]
    if not literal_in_question(q, entry["where_val"], coltype):
        fails.append(f"리터럴 미포함 ({entry['where_val']!r})")
    tokens = set(re.findall(r"[a-z]+", q))
    if tokens & NUMBER_WORDS:
        fails.append(f"숫자를 단어로 씀 ({sorted(tokens & NUMBER_WORDS)})")
    snake_case = re.findall(r"[a-z]+_[a-z]+(?:_[a-z]+)*", q)
    if snake_case:
        fails.append(f"SQL 컬럼명을 그대로 씀, 자연어로 안 풀어씀 ({snake_case})")
    for banned, correct in KNOWN_SYNONYM_SWAPS.items():
        # \b 경계 없이 'in' 으로만 검사하면 "snowshoes" 안의 "shoes"처럼 단어 일부가
        # 우연히 겹치는 경우를 오탐한다 (실제 있었던 버그).
        if re.search(rf"\b{re.escape(banned)}\b", q) and str(entry["where_val"]).lower() != banned:
            fails.append(f"동의어/상위어 치환 의심 ('{banned}' 사용, 정답 리터럴은 '{correct}' 계열)")
    return fails


def soft_flags(q: str, entry: dict) -> list[str]:
    """하드 실패는 아니지만 사람이 확인해야 할 항목 (stock/quantity 혼동 등)."""
    flags = []
    if entry["where_col"] == "stock" or entry["select"] == "stock":
        if "order" in q and "stock" not in q and "left" not in q and "remain" not in q:
            flags.append("stock 질문인데 order 표현만 있음 (quantity와 혼동 의심)")
    if entry["where_col"] == "quantity" or entry["select"] == "quantity":
        if "stock" in q:
            flags.append("quantity 질문인데 stock 언급 (혼동 의심)")
    return flags


def _load_pilot_sql(out_dir: str) -> dict[int, dict]:
    with open(os.path.join(out_dir, "pilot_sql.json"), encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def _load_responses(out_dir: str) -> tuple[dict[int, list[str]], list[str]]:
    resp_dir = os.path.join(out_dir, "responses")
    responses: dict[int, list[str]] = {}
    missing_files = []
    for fname in sorted(os.listdir(resp_dir)) if os.path.isdir(resp_dir) else []:
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(resp_dir, fname), encoding="utf-8") as f:
            try:
                batch = json.load(f)
            except json.JSONDecodeError as e:
                missing_files.append(f"{fname}: JSON 파싱 실패 ({e})")
                continue
        for item in batch:
            responses[int(item["id"])] = item["questions"]
    return responses, missing_files


def validate(out_dir: str) -> dict:
    pilot_sql = _load_pilot_sql(out_dir)
    responses, missing_files = _load_responses(out_dir)

    total_q = 0
    passed_pairs = []
    failures = []
    soft_flag_list = []

    for qid, entry in pilot_sql.items():
        qs = responses.get(qid)
        if qs is None:
            failures.append({"id": qid, "sql": entry["sql"], "question": None, "reasons": ["응답 없음"]})
            continue
        if len(qs) != 5:
            failures.append({"id": qid, "sql": entry["sql"], "question": None,
                              "reasons": [f"질문 5개가 아니라 {len(qs)}개"]})
        for q in qs:
            total_q += 1
            fails = check_question(q, entry)
            if fails:
                failures.append({"id": qid, "sql": entry["sql"], "question": q, "reasons": fails})
            else:
                passed_pairs.append({"question": q, "sql": entry["sql"]})
                sf = soft_flags(q, entry)
                if sf:
                    soft_flag_list.append({"id": qid, "sql": entry["sql"], "question": q, "flags": sf})

    report = {
        "n_sql": len(pilot_sql),
        "n_questions_seen": total_q,
        "n_passed": len(passed_pairs),
        "pass_rate": round(len(passed_pairs) / total_q, 4) if total_q else 0.0,
        "n_failures": len(failures),
        "n_missing_response_files": len(missing_files),
        "missing_response_files": missing_files,
        "n_soft_flags": len(soft_flag_list),
        "failures_sample": failures[:30],
        "soft_flags_sample": soft_flag_list[:30],
    }

    with open(os.path.join(out_dir, "pilot_train_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(passed_pairs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------
# 4. 템플릿(기계적 생성) 탐지
#
# check_question()은 규칙 위반만 잡아낸다. "같은 문장틀에 값만 바꿔 끼웠는가"는
# 규칙 위반이 아니라서 못 잡는다 (LLM이 실제로 다양하게 썼는지 vs 코드/템플릿으로
# 기계적으로 조합했는지는 별도로 확인해야 한다).
# ---------------------------------------------------------------------------

def skeletonize(q: str, entry: dict) -> str:
    """질문에서 리터럴 값을 <V>로 치환한 '틀'. 같은 (select, where_col) 조합에서
    이 틀이 다른 id에도 그대로 재사용되면 값만 바꿔 끼운 템플릿일 가능성이 크다."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sql_gen
    coltype = sql_gen.COLUMN_TYPES[(entry["table"], entry["where_col"])]
    where_val = entry["where_val"]
    if coltype == "text":
        lit = str(where_val).lower()
    elif coltype == "float":
        lit = f"{where_val:.2f}"
    else:
        lit = str(where_val)
    skeleton = q.replace(lit, "<V>")
    if coltype == "float":
        stripped = lit.rstrip("0").rstrip(".")
        if stripped and stripped != lit:
            skeleton = skeleton.replace(stripped, "<V>")
    return skeleton


def template_check(out_dir: str, threshold: float = 0.3) -> dict:
    """(select, where_col) 조합 단위로, 같은 틀(skeleton)이 다른 id에도 재사용된
    비율을 계산한다. 비율이 높으면 LLM이 직접 쓴 게 아니라 템플릿/스크립트로
    기계적으로 채웠다는 신호다."""
    pilot_sql = _load_pilot_sql(out_dir)
    responses, _ = _load_responses(out_dir)

    groups: dict[tuple[str, str], dict[int, list[str]]] = {}
    for qid, entry in pilot_sql.items():
        qs = responses.get(qid)
        if not qs:
            continue
        skeletons = [skeletonize(q, entry) for q in qs]
        groups.setdefault((entry["select"], entry["where_col"]), {})[qid] = skeletons

    shape_reports = []
    total_q = total_repeat = 0
    for shape, by_id in groups.items():
        if len(by_id) < 2:
            continue
        seen: dict[str, int] = {}
        n_q = n_repeat = 0
        examples = []
        for qid, skeletons in by_id.items():
            for sk in skeletons:
                n_q += 1
                if sk in seen and seen[sk] != qid:
                    n_repeat += 1
                    if len(examples) < 3:
                        examples.append(sk)
                else:
                    seen[sk] = qid
        ratio = n_repeat / n_q if n_q else 0.0
        total_q += n_q
        total_repeat += n_repeat
        shape_reports.append({
            "select": shape[0], "where_col": shape[1], "n_ids": len(by_id),
            "n_questions": n_q, "n_repeated_skeleton": n_repeat,
            "repeat_ratio": round(ratio, 3), "example_skeletons": examples,
        })

    shape_reports.sort(key=lambda r: -r["repeat_ratio"])
    flagged = [r for r in shape_reports if r["repeat_ratio"] >= threshold]
    report = {
        "threshold": threshold,
        "overall_repeat_ratio": round(total_repeat / total_q, 3) if total_q else 0.0,
        "n_shapes_checked": len(shape_reports),
        "n_shapes_flagged": len(flagged),
        "flagged_shapes": flagged,
        "all_shapes": shape_reports,
    }
    with open(os.path.join(out_dir, "template_check_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="1단계 LLM 질문 생성 파이프라인 (수동 호출)")
    ap.add_argument("command", choices=["make-prompts", "validate", "template-check"])
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--train", default="data/sql_train.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-dir", default=None,
                     help="이 디렉터리들의 pilot_sql.json에 있는 SQL은 샘플링 후보에서 제외 (쉼표로 여러 개 지정 가능)")
    ap.add_argument("--threshold", type=float, default=0.3,
                     help="template-check: 이 비율 이상 틀이 재사용되면 그 (select,where_col) 조합을 flag")
    args = ap.parse_args()

    if args.command == "make-prompts":
        exclude_dirs = args.exclude_dir.split(",") if args.exclude_dir else None
        make_prompts(args.out, args.train, args.n, args.batch_size, args.seed, exclude_dirs)
        return 0

    if args.command == "template-check":
        report = template_check(args.out, args.threshold)
        print(f"전체 틀(skeleton) 재사용 비율: {report['overall_repeat_ratio']:.1%}")
        print(f"확인한 (select,where_col) 조합 {report['n_shapes_checked']}개 중 "
              f"{report['n_shapes_flagged']}개가 재사용 비율 {args.threshold:.0%} 이상")
        for r in report["flagged_shapes"][:10]:
            print(f"  [{r['select']:<10} / {r['where_col']:<12}] id {r['n_ids']:>3}개, "
                  f"재사용 {r['repeat_ratio']:.0%}  예: {r['example_skeletons'][:1]}")
        print(f"저장: {os.path.join(args.out, 'template_check_report.json')}")
        return 0

    report = validate(args.out)
    print(f"질문 {report['n_questions_seen']}개 중 {report['n_passed']}개 통과 "
          f"(통과율 {report['pass_rate']:.1%}, 목표 90%)")
    print(f"실패 {report['n_failures']}건, 응답 파일 누락 {report['n_missing_response_files']}건, "
          f"수동확인 flag {report['n_soft_flags']}건")
    print(f"저장: {os.path.join(args.out, 'pilot_train_pairs.json')}")
    print(f"      {os.path.join(args.out, 'validation_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
