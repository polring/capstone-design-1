"""
Text-to-SQL 1단계 LLM 질문 생성 파이프라인 (v1, 수동 호출 모드)

sql_gen.py 가 만든 sql_train.json 에서 SQL을 뽑아, 외부 LLM에게 보낼 배치 프롬프트를
텍스트 파일로 만든다. API를 직접 호출하지 않고, 사람이 프롬프트를 챗봇에 붙여넣고
받아온 응답을 다시 이 스크립트로 검증하는 흐름이다.

SQL 정답은 이미 sql_gen.py 가 정했다. LLM은 그 SQL에 맞는 자연어 질문만 만들고,
질문이 규칙(리터럴 포함, 소문자, 숫자 단어 금지 등)을 지키는지는 항상 코드가 검사한다
(CLAUDE.md 스크래치 원칙: LLM은 SQL 정답을 만들거나 수정하지 않는다).

사용법
    python src/data/generator/question_gen.py make-prompts --out data/pilot --n 200 --batch-size 25 --seed 0
        data/pilot/prompts/batch_XX.txt          LLM에 붙여넣을 프롬프트
        data/pilot/pilot_sql.json                배치별 SQL과 메타데이터 (검증용)

    python src/data/generator/question_gen.py make-prompts-small --out data/pilot_small --n 200 --seed 0
        make-prompts와 동일하지만 14B급 이하 약한 모델용 프롬프트를 만든다:
        배치 크기가 작고(기본 8, --batch-size로 조절), 규칙을 더 구체적으로 풀어 쓰고,
        5문장의 말투 슬롯을 명시적으로 지정하고, few-shot 예시를 포함한다.

    (사람이 각 batch_XX.txt 를 LLM에 붙여넣고, 응답 JSON을
     data/pilot/responses/batch_XX.json 으로 저장)

    python src/data/generator/question_gen.py validate --out data/pilot
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
10. if select and where_col are the SAME column, the sql's answer already equals a value stated in the
    question -- do NOT write a redundant "what is X of the thing whose X is <value>" question (e.g. never
    "what price is the item priced 97.56?"). Instead phrase all 5 as an existence/confirmation check, e.g.
    "is there an item priced at 97.56?", "confirm the item priced 97.56 exists", "check if a customer with
    id 1653 exists", "does an order with id 5432 exist".

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


QUESTION_STYLE_SLOTS = [
    "casual/conversational style",
    "formal, complete-sentence style",
    "terse keyword-only style (no full sentence, just the key words)",
    'a direct command ("show me...", "give me...", "look up...")',
    'a "how many / what / which / is there" question',
    "a very short, telegraphic style (fewest words possible)",
    'an informal style like a text message to a coworker (may drop "the"/"a")',
    'a polite request style ("would you mind...", "could you possibly...")',
]


EXAMPLE_QUESTIONS_BY_SLOT = [
    "yo what city is customer 1042 from",
    "could you please tell me which city customer id 1042 lives in?",
    "customer 1042 city",
    "look up the city for customer id 1042.",
    "which city is customer 1042 in?",
    "1042 city?",
    "customer 1042 city pls",
    "would you mind telling me the city for customer id 1042?",
]


def small_model_rules_blurb(n_questions: int = 5) -> str:
    if n_questions > len(QUESTION_STYLE_SLOTS):
        raise ValueError(f"n_questions={n_questions}는 정의된 말투 슬롯({len(QUESTION_STYLE_SLOTS)}개)보다 많음")
    slots = "\n".join(f"{i}. {s}" for i, s in enumerate(QUESTION_STYLE_SLOTS[:n_questions], start=1))
    example_lines = ",\n    ".join(f'"{q}"' for q in EXAMPLE_QUESTIONS_BY_SLOT[:n_questions])
    return f"""\
For EACH sql below, write {n_questions} different english questions that a person could ask to get exactly
that sql as the answer.

Write ONE question for EACH of these {n_questions} slots below, in this exact order. Do not skip a slot or
repeat a style.
{slots}

Rules (all must hold for EVERY question):
1. all lowercase. no capital letters anywhere.
2. the question MUST contain the where-value EXACTLY as given, character for character.
   example: value "marilyn" -> the word "marilyn" must appear. value 1864 -> the digits "1864" must appear.
3. never replace an item/category word with a different word. write it exactly as given.
   wrong: value is "laptop" but question says "computer". wrong: value is "sneakers" but question says "shoes".
4. always write numbers as digits, never spelled out. wrong: "twenty". right: "20".
5. only use facts from THIS one sql line. never assume information from another table.
6. if select is "stock": ask about inventory / units left. if select is "quantity": ask about how many units
   were ordered. these are two different columns -- do not mix them up.
7. if select is "*": ask a general "tell me about" / "show me everything about" question. do not name one
   specific column (not "what is the price of...", just "tell me about...").
8. never write a column name with an underscore (item_id, order_id, customer_id, item_name) in the question.
   write it as separate natural words instead: "item id", "order", "customer", "item".
9. if select and where_col are the SAME column: do NOT ask "what is x of the thing whose x is <value>".
   instead write an existence-check question: "is there...", "confirm...", "check if...", "does... exist".

Worked example (a DIFFERENT sql, only to show the exact output shape -- do not reuse these words):
sql: SELECT city FROM customers WHERE customer_id = 1042
[
  {{"id": 999, "questions": [
    {example_lines}
  ]}}
]

Return ONLY a JSON array in exactly the shape shown above, one object per sql below. Nothing else.
- do not add ```json or ``` code fences.
- do not add any explanation, note, or text before or after the JSON array.
- do not wrap the array in another object.
"""


def build_small_model_batch_prompt(batch: list[dict], n_questions: int = 5) -> str:
    """build_batch_prompt()와 같은 정보를 담되, 14B급 이하 모델을 겨냥해 규칙을 더 구체적으로 풀어
    쓰고, n_questions개 문장의 각 말투 슬롯을 명시적으로 지정하고, few-shot 예시 1개를 포함한 프롬프트."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sql_gen

    lines = [SCHEMA_BLURB, "", small_model_rules_blurb(n_questions), "", "sql list:"]
    for item in batch:
        e = item["entry"]
        coltype = sql_gen.COLUMN_TYPES[(e["table"], e["where_col"])]
        lit = format_literal_for_prompt(e["where_val"], coltype)
        lines.append(
            f'- id={item["id"]} | sql: {e["sql"]} | select={e["select"]} | '
            f'where_col={e["where_col"]} | where_value={lit} | exists={not e["is_nonexistent"]}'
        )
    return "\n".join(lines)


def _make_prompts_impl(out_dir: str, train_path: str, n: int, batch_size: int, seed: int,
                        exclude_dirs: list[str] | None, prompt_builder, log_prefix: str = "") -> None:
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
        prompt = prompt_builder(batch)
        path = os.path.join(out_dir, "prompts", f"batch_{bi:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt)

    pilot_sql = {item["id"]: item["entry"] for item in items}
    with open(os.path.join(out_dir, "pilot_sql.json"), "w", encoding="utf-8") as f:
        json.dump(pilot_sql, f, ensure_ascii=False, indent=2)

    print(f"{log_prefix}SQL {len(items)}개 -> {len(batches)}개 배치 (배치 크기 {batch_size})")
    print(f"프롬프트: {os.path.join(out_dir, 'prompts')}/batch_01.txt ~ batch_{len(batches):02d}.txt")
    print(f"응답을 저장할 위치: {os.path.join(out_dir, 'responses')}/batch_01.json ~ batch_{len(batches):02d}.json")
    print(f"메타데이터: {os.path.join(out_dir, 'pilot_sql.json')}")


def make_prompts(out_dir: str, train_path: str, n: int, batch_size: int, seed: int,
                  exclude_dirs: list[str] | None = None) -> None:
    _make_prompts_impl(out_dir, train_path, n, batch_size, seed, exclude_dirs, build_batch_prompt)


def make_prompts_small_model(out_dir: str, train_path: str, n: int, batch_size: int, seed: int,
                              exclude_dirs: list[str] | None = None, n_questions: int = 5) -> None:
    """14B급 이하 약한 모델용. make_prompts()와의 차이:
    - 배치 크기를 작게(기본 8) 잡아 한 프롬프트가 다뤄야 할 SQL 개수를 줄인다 -- 배치가 커질수록
      표현이 "무난한 패턴"으로 수렴하는 문제가 약한 모델에서 더 빨리, 더 심하게 나타나기 때문이다.
    - small_model_rules_blurb(더 구체적인 규칙 설명 + n_questions개 문장 슬롯 명시 + few-shot 예시)를 쓴다.
    - n_questions로 SQL 하나당 요구하는 질문 개수를 조절할 수 있다(기본 5, QUESTION_STYLE_SLOTS 길이인
      8까지 가능).
    """
    import functools
    builder = functools.partial(build_small_model_batch_prompt, n_questions=n_questions)
    _make_prompts_impl(out_dir, train_path, n, batch_size, seed, exclude_dirs, builder,
                        log_prefix=f"[작은 모델용, {n_questions}문장] ")


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


def _isolated_number_match(q: str, lit: str) -> bool:
    """숫자 리터럴이 더 긴 숫자(정수부가 늘어나거나 소수부가 이어지는 경우) 안에
    파묻혀 우연히 매치되는 것만 막는다. 문장을 끝내는 마침표처럼 뒤에 숫자가
    더 없는 '.'는 정상 통과시킨다 (예: 목표값 97.56이 "...97.56." 끝에 와도 통과,
    197.56 안에 파묻히거나 97.567 처럼 소수부가 더 이어지면 차단)."""
    for m in re.finditer(re.escape(lit), q):
        start, end = m.start(), m.end()
        before_ok = not (start > 0 and (
            q[start - 1].isdigit()
            or (q[start - 1] == "." and start >= 2 and q[start - 2].isdigit())
        ))
        after_ok = not (end < len(q) and (
            q[end].isdigit()
            or (q[end] == "." and end + 1 < len(q) and q[end + 1].isdigit())
        ))
        if before_ok and after_ok:
            return True
    return False


def literal_in_question(q: str, where_val, coltype: str) -> bool:
    if coltype == "text":
        # 뒤에 자연스러운 복수형 어미('s, es, s)가 붙는 것은 허용하되 (예: colander -> colanders),
        # 완전히 다른 단어 안에 파묻히는 것(예: ana가 anastasia 안에 들어있는 경우)은 \b로 차단한다.
        lit = re.escape(str(where_val).lower())
        return re.search(rf"\b{lit}(?:'?s|es)?\b", q) is not None
    if coltype == "float":
        full = f"{where_val:.2f}"
        if _isolated_number_match(q, full):
            return True
        stripped = full.rstrip("0").rstrip(".")
        return bool(stripped and stripped != full and _isolated_number_match(q, stripped))
    return _isolated_number_match(q, str(where_val))


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
    if entry["select"] == entry["where_col"]:
        existence_markers = ("is there", "exist", "confirm", "verify", "check")
        if not any(m in q for m in existence_markers):
            flags.append("select==where_col인데 존재확인/검증 형태가 아님 (동어반복형 의심, 규칙 10 위반)")
    return flags


def _normalize_for_dup(q: str) -> str:
    """중복 판정용 정규화: 문장부호 제거 + 공백 정리. 완전 동일 문장뿐 아니라
    문장부호만 다른 사실상 동일 문장도 같은 키로 묶는다."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q)).strip()


def find_intra_group_duplicates(qs: list[str]) -> set[int]:
    """같은 id(=같은 SQL)의 5문장 안에서, 먼저 나온 문장과 사실상 동일한
    (정규화 후 일치) 문장의 인덱스를 반환한다 (규칙 6: 5문장은 서로 달라야 함).
    첫 등장은 정상으로 두고, 재등장한 것만 실패 처리한다."""
    seen: set[str] = set()
    dup_idx: set[int] = set()
    for i, q in enumerate(qs):
        norm = _normalize_for_dup(q)
        if norm in seen:
            dup_idx.add(i)
        else:
            seen.add(norm)
    return dup_idx


def find_cross_sql_question_conflicts(pairs: list[dict]) -> list[dict]:
    """전체 코퍼스에서 동일한 질문 문자열이 서로 다른 SQL에 매핑된 경우를 찾는다.
    같은 입력(질문)에 다른 정답(SQL)이 붙는 것은 라벨 충돌이라 학습에 해롭다."""
    by_question: dict[str, set[str]] = {}
    for p in pairs:
        by_question.setdefault(p["question"], set()).add(p["sql"])
    conflicts = []
    for q, sqls in by_question.items():
        if len(sqls) > 1:
            conflicts.append({"question": q, "conflicting_sqls": sorted(sqls)})
    return conflicts


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
        if not isinstance(batch, list):
            missing_files.append(f"{fname}: 응답이 JSON 배열이 아님 ({type(batch).__name__})")
            continue
        for item in batch:
            try:
                qid = int(item["id"])
                qs = item["questions"]
            except (KeyError, TypeError, ValueError) as e:
                missing_files.append(f"{fname}: 항목 형식 오류 ({e}) - {item!r}")
                continue
            if qid in responses:
                missing_files.append(f"{fname}: id {qid} 중복 등장 (이전 응답을 덮어씀)")
            responses[qid] = qs
    return responses, missing_files


def validate(out_dir: str, expected_questions: int = 5) -> dict:
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
        if len(qs) != expected_questions:
            failures.append({"id": qid, "sql": entry["sql"], "question": None,
                              "reasons": [f"질문 {expected_questions}개가 아니라 {len(qs)}개"]})
        dup_idx = find_intra_group_duplicates(qs)
        for i, q in enumerate(qs):
            total_q += 1
            fails = check_question(q, entry)
            if i in dup_idx:
                fails = fails + ["같은 id 내 중복/사실상 동일 질문 (규칙6 위반)"]
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
# 5. 전역 코퍼스 검사 (라운드를 넘나드는 충돌 탐지)
# ---------------------------------------------------------------------------

def load_all_merged_pairs(data_dir: str) -> list[dict]:
    """지금까지 merge된 모든 라운드(pilot*, eval_indist*)의 (질문, SQL) 쌍을 합쳐서 반환.
    dataset.py의 load_train_pairs/load_eval_indist_pairs와 같은 glob 규칙을 쓴다."""
    import glob
    pairs: list[dict] = []
    for pattern in ("pilot*/pilot_train_pairs.json", "eval_indist*/pilot_train_pairs.json"):
        for pf in sorted(glob.glob(os.path.join(data_dir, pattern))):
            with open(pf, encoding="utf-8") as f:
                pairs.extend(json.load(f))
    return pairs


def corpus_check(data_dir: str) -> dict:
    pairs = load_all_merged_pairs(data_dir)
    conflicts = find_cross_sql_question_conflicts(pairs)
    report = {
        "n_pairs_checked": len(pairs),
        "n_conflicting_questions": len(conflicts),
        "conflicts": conflicts,
    }
    out_path = os.path.join(data_dir, "corpus_check_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="1단계 LLM 질문 생성 파이프라인 (수동 호출)")
    ap.add_argument("command", choices=["make-prompts", "make-prompts-small", "validate",
                                         "template-check", "corpus-check"])
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--train", default="data/sql_train.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=None,
                     help="한 프롬프트에 담을 SQL 개수 (기본값: make-prompts=25, make-prompts-small=8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-dir", default=None,
                     help="이 디렉터리들의 pilot_sql.json에 있는 SQL은 샘플링 후보에서 제외 (쉼표로 여러 개 지정 가능)")
    ap.add_argument("--threshold", type=float, default=0.3,
                     help="template-check: 이 비율 이상 틀이 재사용되면 그 (select,where_col) 조합을 flag")
    ap.add_argument("--data-dir", default="data",
                     help="corpus-check: pilot*/eval_indist* 병합 결과가 있는 상위 디렉터리")
    ap.add_argument("--questions-per-sql", type=int, default=5,
                     help="make-prompts-small: SQL 하나당 요구할 질문 개수 (기본 5, 최대 %d)"
                          % len(QUESTION_STYLE_SLOTS))
    ap.add_argument("--expected-questions", type=int, default=5,
                     help="validate: 항목당 기대하는 질문 개수 (make-prompts-small을 --questions-per-sql로 "
                          "다르게 만들었다면 여기도 맞춰줘야 함)")
    args = ap.parse_args()

    if args.command == "corpus-check":
        report = corpus_check(args.data_dir)
        print(f"전체 병합 쌍 {report['n_pairs_checked']}개 검사")
        print(f"동일 질문이 서로 다른 SQL에 매핑된 충돌: {report['n_conflicting_questions']}건")
        for c in report["conflicts"][:10]:
            print(f"  질문: {c['question']!r}")
            for s in c["conflicting_sqls"]:
                print(f"    -> {s}")
        print(f"저장: {os.path.join(args.data_dir, 'corpus_check_report.json')}")
        return 0

    if args.command == "make-prompts":
        exclude_dirs = args.exclude_dir.split(",") if args.exclude_dir else None
        batch_size = args.batch_size if args.batch_size is not None else 25
        make_prompts(args.out, args.train, args.n, batch_size, args.seed, exclude_dirs)
        return 0

    if args.command == "make-prompts-small":
        exclude_dirs = args.exclude_dir.split(",") if args.exclude_dir else None
        batch_size = args.batch_size if args.batch_size is not None else 8
        make_prompts_small_model(args.out, args.train, args.n, batch_size, args.seed, exclude_dirs,
                                  n_questions=args.questions_per_sql)
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

    report = validate(args.out, expected_questions=args.expected_questions)
    print(f"질문 {report['n_questions_seen']}개 중 {report['n_passed']}개 통과 "
          f"(통과율 {report['pass_rate']:.1%}, 목표 90%)")
    print(f"실패 {report['n_failures']}건, 응답 파일 누락 {report['n_missing_response_files']}건, "
          f"수동확인 flag {report['n_soft_flags']}건")
    print(f"저장: {os.path.join(args.out, 'pilot_train_pairs.json')}")
    print(f"      {os.path.join(args.out, 'validation_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
