"""
demo.py — 파이프라인 각 단계의 입력 → 출력을 직접 보여 준다 (확인·시연용, 파일을 쓰지 않는다)

저장소 루트에서 실행한다. 읽는 파일은 모두 인자로 바꿀 수 있고, 생략하면 src/config.py 기본값을 쓴다.

    python -m src.demo db --sql "SELECT city FROM customers WHERE name = 'ashley'"   # 이 SQL 이 읽는 행
    python -m src.demo sql --sql "..."            # 이 SQL 의 생성 정보 (어느 분할에 있는지, table·where_col·값)
    python -m src.demo questions --sql "..."      # 이 SQL 에 LLM 이 쓴 질문들과 검증 결과
    python -m src.demo clean --text "..." --sql "..."       # 이 (질문, SQL) 의 라벨 규칙 판정
    python -m src.demo name-swap --text "..." --sql "..."   # 이 쌍을 가짜 이름으로 바꾼 결과
    python -m src.demo dataset --text "..." --sql "..."     # (질문, SQL) → 학습 입력 ID·loss_mask, 전체 길이 통계
    python -m src.demo tokenizer --text "..."     # 텍스트 → 토큰 → 텍스트
    python -m src.demo embedding --text "..."     # 토큰 ID → 임베딩, RoPE
    python -m src.demo block --text "..."         # 디코더 블록 내부 단계별 계산
    python -m src.demo model --text "..." --sql "..."       # 최종 출력층, 다음 토큰 확률, loss
    python -m src.demo generate --text "..."      # greedy 생성을 한 토큰씩
    python -m src.demo score --text "..." --sql "..."       # 생성 SQL 을 정답과 채점 (EM, 다중 정답, 실행 정확도, 틀린 부분)
    python -m src.demo all --sql "..."            # 예시 하나를 위 단계 전체로 따라간다 (앞 단계 결과를 다음 단계에 넘김)

- 하위 명령은 각자 독립적으로 돈다. --sql / --text 를 주면 그 예시를, 주지 않으면 하위 명령마다 정해진 기준으로
  고른다 (python -m src.demo --help, docs/SETUP.md 9절).
- all 은 예시 하나를 처음부터 끝까지 따라간다: SQL → DB 행 → 생성 정보 → LLM 질문 → 라벨 판정 → 이름 교체 →
  학습 입력 → 토큰 → 임베딩 → 디코더 블록 → 출력층 → 생성 SQL → 채점. 각 단계의 결과가 다음 단계의 입력이 된다.
- 출력의 ※ 줄은 그 단계가 무엇을 하는지, 나온 값을 어떻게 읽는지 설명한다. --quiet 를 주면 값만 출력한다.
- 질문을 SQL 로 바꿔 쓰거나 평가하는 것은 src/run.py 다. 모델 단계는 배포 모델(models/stage<N>/model.pt)을 CPU 에서
  쓰고(--model 로 변경), 데이터 단계는 1단계 코드(src/data/stage1/)를 쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data.dataset import (BOS_ID, EOS_ID, SEP_ID, TextToSQLDataset, collate_fn, load_eval_pairs,
                              load_train_pairs, tokenize_pairs)
from src.evaluation.eval_sets import list_sets
from src.evaluation.scoring import execute, load_rules
from src.inference.generate import predict_questions, safe_decode
from src.inference.loading import load_model, tokenizer_path_for
from src.models.embedding import apply_rotary_emb
from src.tokenizer import bpe

DEFAULT_TEXT = "what city does ashley live in?"
DEFAULT_SQL = "SELECT city FROM customers WHERE name = 'ashley'"
SQL_FILES = {"train": config.SQL_TRAIN_PATH, "eval_indist": config.SQL_EVAL_INDIST_PATH,
             "eval_holdout": config.SQL_EVAL_HOLDOUT_PATH}
CONFUSED_P = 0.01   # generate: 2위 후보 확률이 이 이상이면 "모델이 헷갈린 토큰"으로 표시
QUIET = False       # --quiet: 설명(※) 줄을 끄고 값만 출력

SPLIT_MEANING = {"train": "학습 SQL - 모델이 학습 때 이 SQL 과 그 질문들을 봤다",
                 "eval_indist": "분포 내 평가 SQL - 학습에 없던 SQL 이다 (값·구조는 학습과 같은 분포)",
                 "eval_holdout": "미등장 값 평가 SQL - 값이 학습에 한 번도 나오지 않았다"}
TIER_MEANING = {"unseen_entity": "처음 보는 이름·상품명과 그 ID", "unseen_id": "이름은 학습에 있고 ID 만 처음",
                "unseen_order_id": "처음 보는 주문 ID"}
ERROR_PART_MEANING = {"table": "다른 테이블을 골랐다", "where_col": "다른 조건 컬럼을 골랐다",
                      "select": "다른 SELECT 컬럼을 골랐다", "literal": "값을 잘못 옮겨 적었다",
                      "malformed": "1단계 SQL 형식이 아니다"}


# ---------------------------------------------------------------------------
# 출력·로딩 보조
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def section(title: str) -> None:
    print(f"\n--- {title}")


def note(*lines: str) -> None:
    """사용자를 위한 설명 (※ 로 시작). --quiet 이면 출력하지 않는다."""
    if QUIET:
        return
    for i, line in enumerate(lines):
        print(("  ※ " if i == 0 else "    ") + line)


def token_str(tok: bpe.Tokenizer, i: int) -> str:
    """토큰 ID 하나를 사람이 읽을 수 있게: 특수 토큰은 이름, 나머지는 바이트를 문자열로."""
    if bpe.SPECIAL_BASE <= i < bpe.SEED_BASE:
        return bpe.SPECIAL_TOKENS[i - bpe.SPECIAL_BASE]
    if i not in tok.id_to_bytes:
        return f"<unk:{i}>"
    return repr(tok.id_to_bytes[i].decode("utf-8", errors="replace"))


def kind_of(tok: bpe.Tokenizer, i: int) -> str:
    if i < bpe.SPECIAL_BASE:
        return "byte"
    if i < bpe.SEED_BASE:
        return "special"
    return "seed" if i < tok.merge_base else "merge"


def rms(t: torch.Tensor) -> float:
    return t.detach().float().pow(2).mean().sqrt().item()


def stats(t: torch.Tensor) -> str:
    t = t.detach().float()
    return f"shape={tuple(t.shape)}  mean={t.mean():+.4f}  std={t.std():.4f}  rms={rms(t):.4f}"


def load_tokenizer(args, model_path: Path | None = None, ckpt: dict | None = None) -> bpe.Tokenizer:
    """--tokenizer > 모델 폴더의 토크나이저 > config.TOKENIZER_PATH"""
    if getattr(args, "tokenizer", None):
        return bpe.Tokenizer.load(args.tokenizer)
    if model_path is not None:
        return bpe.Tokenizer.load(tokenizer_path_for(model_path, ckpt or {}))
    return bpe.Tokenizer.load(config.TOKENIZER_PATH)


def load_model_and_tokenizer(args):
    model_path = Path(args.model)
    model, ckpt = load_model(model_path, torch.device("cpu"))
    return model, load_tokenizer(args, model_path, ckpt)


def prompt_of(tok: bpe.Tokenizer, question: str) -> list[int]:
    """<bos> 질문 <sep> — 추론 때 모델이 받는 입력 (질문은 학습 데이터처럼 소문자로)."""
    return [BOS_ID] + tok.encode(question.strip().lower()) + [SEP_ID]


def text_of(args) -> str:
    return args.text or DEFAULT_TEXT


def sql_of(args) -> str:
    return args.sql or DEFAULT_SQL


def rows_query(sql: str) -> str:
    """SELECT 부분만 * 로 바꿔, 이 SQL 이 읽는 행 전체를 보는 쿼리."""
    return re.sub(r"^SELECT .+? FROM ", "SELECT * FROM ", sql, count=1)


def find_sql_entry(sql: str) -> tuple[str, dict] | None:
    """SQL 파일들(학습·분포 내·미등장 값)에서 sql 의 생성 정보를 찾는다. (분할 이름, 항목)."""
    for split, path in SQL_FILES.items():
        for r in json.loads(Path(path).read_text(encoding="utf-8")):
            if r["sql"] == sql:
                return split, r
    return None


# ---------------------------------------------------------------------------
# 데이터 단계 (각 함수는 결과를 반환해 all 에서 다음 단계 입력으로 쓴다)
# ---------------------------------------------------------------------------

def demo_db(args, sql: str | None = None) -> list[tuple]:
    """sql 을 주면 그 SQL 이 읽는 테이블의 구조와 조건에 맞는 행, 없으면 테이블마다 앞쪽 --n 행."""
    header(f"DB  (db_gen 출력)  입력: {args.db}")
    note("shop.db 는 코드(db_gen)가 시드로 만든 쇼핑몰 DB 다 (고객·상품·주문 3개 테이블). 시드가 같으면 항상 같은 DB 가",
         "나온다. 이후 모든 SQL 의 정답과 실행 결과는 이 DB 에서 확인한다.")
    con = sqlite3.connect(args.db)
    tables = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    if sql:
        tables = [re.search(r" FROM (\w+)", sql).group(1)]
    rows: list[tuple] = []
    for table in tables:
        cols = [(r[1], r[2]) for r in con.execute(f"PRAGMA table_info({table})")]
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        section(f"{table}  ({n}행)  컬럼: " + ", ".join(f"{c} {t}" for c, t in cols))
        query = rows_query(sql) if sql else f"SELECT * FROM {table} LIMIT {args.n}"
        rows = con.execute(query).fetchall()
        if sql:
            print(f"  입력 SQL 의 WHERE 조건에 맞는 행 ({query}): {len(rows)}행")
        for row in rows[:max(args.n, 5)]:
            print("   ", " | ".join(str(v) for v in row))
        if sql:
            note("입력 SQL 은 이 행들을 읽는다. 정답 실행 결과는 이 행에서 SELECT 컬럼만 뽑은 것이다."
                 + (" 0행이면 없는 값을 묻는 SQL 이고, 그때 정답 결과는 빈 결과다." if not rows else ""))
    con.close()
    if not sql:
        holdout = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
        section(f"미등장 값 ({args.holdout}): 학습 SQL 에 절대 나오지 않는 값")
        for table, info in holdout.items():
            if isinstance(info, dict) and "heldout_rows" in info:
                print(f"  {table}: {len(info['heldout_rows'])}개  예: {info['heldout_rows'][:2]}")
        note("이 값들은 DB 에는 있지만 학습 SQL 에서는 빼 두고 평가에서만 쓴다. 모델이 처음 보는 이름·ID 를 외운 목록이",
             "아니라 질문에서 복사해 SQL 을 만드는지 잰다.")
    return rows


def demo_sql(args, sql: str | None = None) -> dict | None:
    """sql 을 주면 SQL 파일들에서 그 SQL 의 생성 정보를 찾아 보여 주고 반환한다. 없으면 --sql-file 에서 무작위 --n 개."""
    header("SQL 생성  (sql_gen 출력)")
    note("정답 SQL 은 LLM 이 아니라 코드가 만든다. (테이블, WHERE 컬럼) 조합마다 DB 에 있는 값을 골라",
         "1단계 문법 SELECT <컬럼 또는 *> FROM <테이블> WHERE <컬럼> = <값> 으로 만들고, 학습·평가는 SQL 단위로 나눈다.")
    con = sqlite3.connect(args.db)
    if sql:
        found = find_sql_entry(sql)
        section(f"입력 SQL: {sql}")
        if found is None:
            print("  생성된 SQL 목록(sql/*.json)에 없는 SQL 이다 (직접 쓴 SQL).")
        else:
            split, r = found
            print(f"  분할: {split}  ({SQL_FILES[split]})")
            print(f"  생성 정보: table={r['table']}  select={r['select']}  where_col={r['where_col']}  "
                  f"where_val={r['where_val']!r}" + (f"  tier={r['holdout_tier']}" if "holdout_tier" in r else ""))
            note(SPLIT_MEANING[split] + "."
                 + (f" 계층 {r['holdout_tier']}: {TIER_MEANING.get(r['holdout_tier'], '')}." if "holdout_tier" in r else ""))
            if r.get("is_nonexistent"):
                note("DB 에 없는 ID 를 묻는 SQL 이라 정답 결과가 0행이다. ID 의 첫 자리만 보고 테이블을 맞히는 지름길을",
                     "막으려고 일부러 섞었다.")
        res = con.execute(sql).fetchall()
        print(f"  실행 결과(정답): {len(res)}행 {res[:3]}{' ...' if len(res) > 3 else ''}")
        note("다음 단계(questions): LLM 이 이 SQL 에 맞는 영어 질문을 쓴다.")
        con.close()
        return found[1] if found else None
    rows = json.loads(Path(args.sql_file).read_text(encoding="utf-8"))
    print(f"입력: {args.sql_file} 의 SQL {len(rows)}개 중 무작위 {args.n}개 (seed {args.seed})")
    for r in random.Random(args.seed).sample(rows, min(args.n, len(rows))):
        section(f"입력: table={r['table']}  select={r['select']}  where_col={r['where_col']}  where_val={r['where_val']!r}")
        print(f"  출력 SQL : {r['sql']}")
        res = con.execute(r["sql"]).fetchall()
        print(f"  실행 결과: {len(res)}행 {res[:3]}{' ...' if len(res) > 3 else ''}")
    con.close()
    return None


def _pair_files(args) -> list[Path]:
    """질문 쌍을 찾을 파일: --pairs, 없으면 학습 폴더 *.json + 평가셋 폴더의 pairs.json 전부."""
    if args.pairs:
        return [Path(p) for p in args.pairs]
    files = sorted(Path(config.TRAIN_DIR).glob("*.json"))
    return files + [Path(args.eval_dir) / n / config.PAIRS_FILE for n in list_sets(args.eval_dir)]


def demo_questions(args, sql: str | None = None, entry: dict | None = None) -> list[str]:
    """sql 을 주면 질문 쌍 파일들에서 그 SQL 의 질문을 모두 찾아 검증 결과와 함께 보여 주고 반환한다.
    없으면 질문이 있는 SQL 무작위 --n 개."""
    from src.data.stage1 import question_gen as qg
    header("질문 생성  (question_gen 검증: LLM 이 쓴 질문이 규칙을 지키는지 코드가 검사)")
    note("LLM 은 SQL 을 보고 질문만 쓴다 (SQL 정답은 고치지 않는다). 코드가 규칙을 검사해 통과한 질문만 쓴다:",
         "SQL 의 값이 질문에 그대로 들어 있는지, 소문자인지, 숫자를 단어로 풀어 쓰지 않았는지 등.",
         "[ ] 안은 그 질문이 들어 있는 파일이다 (학습 파일 이름 또는 평가셋 이름).")
    files = _pair_files(args)
    by_sql: dict[str, list[tuple[str, str, bool]]] = {}
    for f in files:
        source = f.parent.name if f.name == config.PAIRS_FILE else f.name   # 평가셋 이름 또는 학습 파일 이름
        is_train = Path(config.TRAIN_DIR).resolve() in f.resolve().parents
        for p in json.loads(f.read_text(encoding="utf-8")):
            by_sql.setdefault(p["sql"], []).append((p["question"], source, is_train))
    if not sql:
        entries = {r["sql"]: r for r in json.loads(Path(args.sql_file).read_text(encoding="utf-8"))}
        cands = [s for s in by_sql if s in entries]
        print(f"입력: {args.sql_file} 의 SQL 중 질문이 있는 것에서 무작위 {args.n}개 (seed {args.seed})")
        for s in random.Random(args.seed).sample(cands, min(args.n, len(cands))):
            _print_questions(qg, s, entries[s], by_sql[s])
        return []
    if entry is None:
        found = find_sql_entry(sql)
        entry = found[1] if found else None
    questions = by_sql.get(sql, [])
    _print_questions(qg, sql, entry, questions)
    if not questions:
        print(f"  이 SQL 의 질문이 질문 쌍 파일에 없다 ({len(files)}개 파일 검색).")
    else:
        sources = sorted({src for _, src, _ in questions})
        in_both = len({t for _, _, t in questions}) > 1
        note(f"질문 {len(questions)}개 ({', '.join(sources)}). SQL 하나에 표현이 다른 질문 여러 개를 붙여, 같은 뜻의 다른",
             "말투를 같은 SQL 로 바꾸도록 학습한다. 학습 데이터와 평가셋에 같은 SQL 이 함께 있으면 안 된다"
             + (" - 지금 둘 다에 있다!" if in_both else " (이 SQL 은 한쪽에만 있다)."))
    return [q for q, _, _ in questions]


def _print_questions(qg, sql: str, entry: dict | None, questions: list[tuple[str, str, bool]]) -> None:
    section(f"입력 SQL: {sql}")
    for q, source, _ in questions:
        if entry is None:
            print(f"  출력 질문 [{source}]: {q!r}  (생성 정보가 없어 검증 생략)")
            continue
        fails, flags = qg.check_question(q, entry), qg.soft_flags(q, entry)
        mark = "통과" if not fails else "실패: " + ", ".join(fails)
        print(f"  출력 질문 [{source}]: {q!r}  → {mark}" + (f"  [확인 필요: {', '.join(flags)}]" if flags else ""))
    if any(entry and qg.soft_flags(q, entry) for q, _, _ in questions):
        note("[확인 필요] 는 규칙 위반은 아니지만 사람이 읽어 봐야 하는 표현이다 (예: stock 과 quantity 혼동).")


def demo_clean(args, question: str | None = None, sql: str | None = None) -> str | None:
    """(question, sql) 을 주면 그 쌍의 라벨 규칙 판정을 보여 주고 제거 사유(없으면 None)를 반환한다.
    없으면 학습 쌍 전체의 제거 대상 수와 SELECT 모호 문항 무작위 --n 개의 판정."""
    from src.data.stage1.clean_ambiguous import REASONS, drop_reason, select_label
    from src.data.stage1.sql_rules import parse_sql, select_ambiguous
    header("라벨 정제  (clean_ambiguous: 질문만으로 정답 SQL 이 하나로 정해지는지, 라벨 규칙 L2~L4)")
    note("같은 표현의 질문에 서로 다른 정답이 붙어 있으면 모델은 어느 쪽인지 외울 수밖에 없다. 라벨 규칙(계획서 4-5)으로",
         "'질문만 보고 정답이 하나로 정해지는지' 판정해 어긋난 학습 쌍은 지운다. 평가셋은 지우지 않고 규칙 일치 EM 에서만 뺀다.")
    if question and sql:
        g = parse_sql(sql)
        section(f"입력: {question!r} | {sql}")
        if g is None:
            print("  1단계 SQL 형식이 아니라 판정하지 않는다.")
            return None
        amb = select_ambiguous(question, sql)
        print(f"  SELECT 모호 문항(* 와 식별 컬럼 중 무엇인지 질문에 단서 없음): {amb}"
              + (f"  → 규칙상 SELECT={select_label(question.lower(), g)}" if amb else ""))
        reason = drop_reason({"question": question, "sql": sql})
        print(f"  출력: {'통과 (학습 데이터로 씀)' if reason is None else '제거 - ' + REASONS[reason]}")
        if amb:
            note("'which item ...' 처럼 개체를 묻지만 이름만 원하는지 행 전체를 원하는지 단서가 없는 질문이다. 규칙(L2)이",
                 "질문 표현으로 한쪽을 정하고, 정답 라벨이 그와 같으면 통과다.")
        if reason == "id":
            note("L3: 자기 ID(item_id, customer_id)는 질문이 ID 를 요구할 때만 SELECT 한다.")
        if reason == "table":
            note("L4: customers/items 로도 같은 답이 나오는 orders SQL 은 질문에 주문 단서(order, bought …)가 있어야 한다.")
        return reason
    pairs = load_train_pairs(args.train_files)
    removed = [(p, r) for p in pairs if (r := drop_reason(p))]
    print(f"입력: 학습 쌍 {len(pairs)}개 → 제거 대상 {len(removed)}개" + (" (정제가 끝난 데이터라 0개가 정상)" if not removed else ""))
    for p, r in removed[:args.n]:
        print(f"  [{REASONS[r]}] {p['question']!r} → {p['sql']}")
    amb = [p for p in pairs if select_ambiguous(p["question"], p["sql"])]
    section(f"SELECT 모호 문항 {len(amb)}개 중 무작위 {args.n}개의 판정")
    for p in random.Random(args.seed).sample(amb, min(args.n, len(amb))):
        g = parse_sql(p["sql"])
        print(f"  입력: {p['question']!r}  정답 SELECT={g['select']}  → 규칙상 SELECT={select_label(p['question'].lower(), g)}"
              f"  → {'통과' if not drop_reason(p) else '제거'}")
    note("정답 SELECT 와 규칙상 SELECT 가 같으면 통과다. 학습 데이터를 추가하면 이 정제를 다시 돌린다(--apply).")
    return None


def demo_name_swap(args, question: str | None = None, sql: str | None = None) -> dict | None:
    """(question, sql) 을 주면 그 쌍을 가짜 이름으로 바꾼 결과를 보여 주고 반환한다 (이름 조건 SQL 일 때).
    없으면 학습 쌍 전체에 --ratio 로 적용한 통계와 바뀐 쌍 앞쪽 --n 개."""
    from src.data.stage1.name_swap import NameSwapper, load_english_words
    header("이름 교체  (name_swap: 이름을 외우지 않고 질문에서 복사하도록, 학습 때 epoch 마다 적용)")
    note("학습 데이터의 이름은 DB 에 있는 몇백 개뿐이라, 그대로 학습하면 모델이 이름 목록을 외워 처음 보는 이름을 못 쓴다",
         "(정답률 0~3%). 매 epoch 이름 조건 쌍의 40% 를 질문·SQL 양쪽에서 가짜 이름으로 바꿔 질문에서 복사하게 만든다",
         "(약 88%). 가짜 이름은 기존 이름 철자를 흉내 내 만들고, 실제 영단어·DB 이름·미등장 값은 피한다.")
    pairs = load_train_pairs(args.train_files)
    swapper = NameSwapper(pairs, load_english_words(args.wordlist), args.holdout)
    rng = random.Random(args.seed)
    if question and sql:
        p = swapper.swap([{"question": question, "sql": sql}], 1.0, rng)[0][0]
        section(f"입력: {question!r} | {sql}")
        if not p["swapped"]:
            print("  출력: 바뀌지 않음 (이름 조건 SQL 이 아니거나 질문에서 이름을 찾지 못함)")
            note("이름(name, item_name) = '…' 조건이 있는 쌍만 바꾼다.")
            return None
        print(f"  출력: {p['question']!r} | {p['sql']}")
        note("질문과 SQL 의 이름이 함께 바뀌어 짝이 유지된다. 학습 때는 이렇게 바뀐 쌍도 함께 보고, 추론에는 원래 질문을 쓴다.")
        return p
    out, st = swapper.swap(pairs, args.ratio, rng)
    print(f"입력: 학습 쌍 {len(pairs)}개, 비율 {args.ratio}  →  통계: {st}")
    note("name_pairs 는 이름 조건 쌍 수, swapped_* 는 바뀐 쌍 수, rejected_english 는 실제 영단어라 버린 가짜 이름 수다.")
    for k, p in enumerate([p for p in out if p["swapped"]][:args.n], 1):
        section(f"예시 {k}")
        print(f"  입력: {p['orig_question']!r} | {p['orig_sql']}")
        print(f"  출력: {p['question']!r} | {p['sql']}")
    return None


def demo_dataset(args, tok: bpe.Tokenizer, question: str, sql: str, corpus_stats: bool = True) -> dict:
    """(question, sql) → 학습 입력. 토큰화된 예시를 반환한다. corpus_stats 면 전체 쌍 길이 통계도 (느림)."""
    header("Dataset  (dataset.py: (질문, SQL) → <bos> 질문 <sep> SQL <eos>, loss 는 SQL 위치만)")
    note("모델 입력은 질문과 SQL 을 이어 붙인 토큰열 하나다. 모델은 매 위치에서 '다음 토큰'을 맞히도록 학습한다.")
    ex = tokenize_pairs([{"question": question, "sql": sql}], tok)[0]
    section(f"입력: {question!r} | {sql}")
    print(f"  출력 ids ({len(ex['ids'])}개, SQL 시작 위치 sql_start={ex['sql_start']}):\n    {ex['ids']}")
    batch = collate_fn([ex])
    section("collate_fn: 다음 토큰 예측용으로 한 칸 밀기, loss 는 SQL 위치에서만 (질문은 추론 때 주어지므로)")
    print("  위치  input          → target         loss")
    for t, (a, b, m) in enumerate(zip(batch["input_ids"][0].tolist(), batch["target_ids"][0].tolist(),
                                      batch["loss_mask"][0].tolist())):
        print(f"  {t:>4}  {token_str(tok, a):<14} → {token_str(tok, b):<14} {'O' if m else '.'}")
    n_loss = int(batch["loss_mask"][0].sum())
    note(f"input 의 각 위치는 바로 다음 토큰(target)을 맞혀야 한다. loss 는 O 표시 {n_loss}곳(SQL 토큰 + <eos>)에서만",
         "계산한다. 추론 때 질문은 주어지므로 질문을 예측하는 법은 배울 필요가 없고, <eos> 를 맞혀야 생성을 멈춘다.")
    if corpus_stats:
        train = load_train_pairs(args.train_files)
        evals = load_eval_pairs("indist", args.eval_dir) if "indist" in list_sets(args.eval_dir) else []
        section(f"전체 길이 통계: 학습 {len(train)}쌍 + 분포 내 평가 {len(evals)}쌍")
        lengths = sorted(len(e["ids"]) for e in tokenize_pairs(train + evals, tok))
        n = len(lengths)
        over = sum(L > config.SEQ_LEN for L in lengths)
        print(f"  길이 min={lengths[0]} max={lengths[-1]} mean={sum(lengths) / n:.1f} p50={lengths[n // 2]} "
              f"p99={lengths[int(n * 0.99)]}, 컨텍스트 {config.SEQ_LEN} 초과 {over}개")
        note(f"컨텍스트(최대 길이 {config.SEQ_LEN})를 넘는 예시가 {over}개" + ("라 잘리는 예시가 없다." if not over else " 있다 - 잘려서 학습된다!"))
        loader = torch.utils.data.DataLoader(TextToSQLDataset(tokenize_pairs(train[:256], tok)),
                                             batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
        b = next(iter(loader))
        print(f"  배치 (bs={config.BATCH_SIZE}): input_ids {tuple(b['input_ids'].shape)}, target_ids "
              f"{tuple(b['target_ids'].shape)}, loss_mask {tuple(b['loss_mask'].shape)} - 배치 안 최대 길이로 오른쪽 패딩")
        note("오른쪽 패딩은 attention mask 없이도 괜찮다: causal attention 은 앞 토큰만 보므로 뒤에 붙은 패딩이 실제 토큰에",
             "영향을 주지 않고, 패딩 위치는 loss 에서 빠진다.")
    return ex


# ---------------------------------------------------------------------------
# 토크나이저·모델 단계
# ---------------------------------------------------------------------------

def demo_tokenizer(args, tok: bpe.Tokenizer, texts: list[str]) -> list[int]:
    """텍스트들을 토큰화해 보여 주고, 첫 텍스트(질문)의 추론 입력 <bos> 질문 <sep> 을 반환한다."""
    header(f"토크나이저  (bpe.py)  vocab {tok.vocab_size} = 바이트 256 + 특수 {len(bpe.SPECIAL_TOKENS)} "
           f"+ seed {len(tok.seed_tokens)} + 병합 {len(tok.merges)}")
    note("바이트 단위 BPE: 텍스트를 바이트로 쪼갠 뒤, 학습 코퍼스에서 자주 붙어 나온 쌍을 병합 규칙 순서대로 합친다.",
         "SQL 키워드·테이블·컬럼 이름·범주형 값(seed 토큰)은 쪼개지 않고 항상 토큰 하나로 둔다.")
    for text in texts:
        section(f"입력: {text!r}")
        pieces = tok.pre_tokenize(text)
        print("  1) 사전 분할 (seed 토큰은 [ ] 로 표시, 통째로 토큰 하나):")
        print("     " + " ".join(f"[{s}]" if sid is not None else repr(s) for s, sid in pieces))
        ids = tok.encode(text)
        print(f"  2) 바이트 → 병합 규칙 적용 → 토큰 ID ({len(ids)}개):")
        for i in ids:
            print(f"     {i:>5}  {kind_of(tok, i):<6} {token_str(tok, i)}")
        print(f"  3) 디코딩: {tok.decode(ids)!r}  (원문과 같음: {tok.decode(ids) == text})")
        split_words = [(s, len(tok.encode(s))) for s, sid in pieces if sid is None and s.strip() and len(tok.encode(s)) > 1]
        if split_words:
            note("여러 토큰으로 나뉜 단어: " + ", ".join(f"{w}({n}개)" for w, n in split_words) + ". 학습에 덜 나온 단어일수록",
                 "잘게 나뉜다. 처음 보는 이름도 이렇게 조각나므로, 모델은 이 조각들을 질문에서 SQL 로 그대로 복사해야 한다.")
        note("디코딩이 원문과 같아야 모델이 만든 토큰을 SQL 문자열로 정확히 되돌릴 수 있다.")
    prompt = prompt_of(tok, texts[0])
    section(f"출력 (다음 단계 입력): <bos> 질문 <sep> = {prompt}")
    note("추론 때 모델은 이 토큰열을 받아 <sep> 뒤에 SQL 을 이어 쓴다.")
    return prompt


def demo_embedding(args, model, tok: bpe.Tokenizer, prompt: list[int]) -> torch.Tensor:
    """토큰 ID → 임베딩. 임베딩 x (1, T, dim) 를 반환한다."""
    header("임베딩 · RoPE  (embedding.py)")
    note("토큰 ID 마다 학습된 256차원 벡터 하나를 꺼낸다 (임베딩 표 1024 × 256). 위치 정보는 여기서 더하지 않고,",
         "각 블록의 attention 안에서 RoPE 로 q·k 를 위치만큼 회전시켜 넣는다.")
    ids = torch.tensor([prompt])
    section(f"입력 토큰 ID {tuple(ids.shape)}: {prompt}")
    with torch.no_grad():
        x = model.tok_emb(ids)
    print(f"  출력 임베딩: {stats(x)}")
    print(f"  첫 토큰(<bos>)의 앞 8차원: {[round(v, 4) for v in x[0, 0, :8].tolist()]}")
    dup = next(((i, j) for i in range(len(prompt)) for j in range(i + 1, len(prompt)) if prompt[i] == prompt[j]), None)
    if dup:
        i, j = dup
        note(f"위치 {i}와 {j}는 같은 토큰 {token_str(tok, prompt[i])} 이라 임베딩도 같다 "
             f"({torch.equal(x[0, i], x[0, j])}). 위치 구분은 RoPE 가 한다.")
    attn = model.blocks[0].attn
    T = ids.size(1)
    freqs = model.freqs_cis[:T]
    section(f"RoPE: 각 블록의 attention 에서 q·k 를 위치마다 회전시킨다. head_dim({attn.head_dim})을 2개씩 묶어 "
            f"freqs_cis {tuple(freqs.shape)} (복소수)만큼 회전")
    print(f"  위치 1의 회전각(앞 4쌍, 라디안): {[round(a, 4) for a in torch.angle(freqs[1, :4]).tolist()]}")
    q = torch.randn(1, T, attn.num_heads, attn.head_dim)
    q_rot, _ = apply_rotary_emb(q, q, freqs)
    print(f"  예: q {tuple(q.shape)} → 회전 후 {tuple(q_rot.shape)}. 회전이라 벡터 크기는 그대로: "
          f"{q.norm():.4f} → {q_rot.norm():.4f}")
    note("차원 쌍마다 회전 속도가 다르다(앞쪽 쌍이 빠름). 두 토큰의 q·k 내적은 두 위치의 '차이'에만 영향을 받아,",
         "모델이 상대 거리(바로 앞 단어, 몇 칸 앞 등)를 쓸 수 있다.")
    section("출력 (다음 단계 입력): 임베딩 x")
    return x


def demo_block(args, model, tok: bpe.Tokenizer, prompt: list[int], x: torch.Tensor) -> torch.Tensor:
    """임베딩 x 를 디코더 블록 전체에 통과시킨다. --layer 블록은 내부를 단계별로 보여 준다. 마지막 블록 출력을 반환한다."""
    header(f"디코더 블록 × {len(model.blocks)}  (transformer.py): 각 블록 x + Attn(RMSNorm(x)), 그다음 x + FFN(RMSNorm(x))")
    note("블록마다 ① attention: 각 토큰이 앞 토큰들을 보고 필요한 정보를 모은다 ② FFN: 토큰마다 따로 변환한다.",
         "둘 다 결과를 입력에 더한다(residual). 블록을 거칠수록 '질문의 어느 부분이 SQL 의 어디로 가는지'가 쌓인다.")
    freqs = model.freqs_cis[:x.size(1)]
    section(f"입력: 임베딩 {stats(x)}")
    rms_trace = [rms(x)]
    with torch.no_grad():
        for li, blk in enumerate(model.blocks):
            if li == args.layer:
                x = _block_detail(blk, tok, prompt, x, freqs, li)
            else:
                x = blk(x, freqs_cis=freqs)
                print(f"  블록 {li} 출력: {stats(x)}")
            rms_trace.append(rms(x))
    section(f"출력 (다음 단계 입력): 마지막 블록 출력 {stats(x)}")
    note("rms(값의 크기)가 블록마다 " + " → ".join(f"{v:.2f}" for v in rms_trace) + " 로 커진다. 블록 출력이 residual 로",
         "계속 더해지기 때문이고 정상이다. 각 블록은 입력을 RMSNorm 으로 맞춘 뒤 쓰고, 출력층 앞에서 최종 RMSNorm 이",
         "다시 크기를 맞춘다.")
    return x


def _block_detail(blk, tok, prompt, x, freqs, li) -> torch.Tensor:
    attn = blk.attn
    T = x.size(1)
    print(f"  블록 {li} 내부 (--layer {li}):")
    h = blk.attn_norm(x)
    print(f"    1) attn_norm (RMSNorm): {stats(h)}  ← 토큰마다 rms 를 1 근처로 맞춘 뒤 weight 를 곱함")
    q, k, v = attn.qkv_proj(h).chunk(3, dim=-1)
    shape = (1, T, attn.num_heads, attn.head_dim)
    q, k, v = q.view(shape), k.view(shape), v.view(shape)
    print(f"    2) qkv_proj → q, k, v 각각 {shape} (헤드 {attn.num_heads}개 × {attn.head_dim}차원)")
    q, k = apply_rotary_emb(q, k, freqs)
    print("    3) q, k 에 RoPE 적용 (위치 정보)")
    scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) / math.sqrt(attn.head_dim)
    scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    w = scores.softmax(-1)
    print(f"    4) attention 가중치 {tuple(w.shape)} (causal: 앞 토큰만 봄). 마지막 토큰(<sep>)이 많이 보는 토큰 (헤드 0):")
    top = w[0, 0, -1].topk(min(5, T))
    for p, j in zip(top.values.tolist(), top.indices.tolist()):
        print(f"         위치 {j:>2} {token_str(tok, prompt[j]):<14} {p:.3f}")
    note("q(찾는 것)와 k(가진 것)의 내적을 softmax 해 '어느 앞 토큰을 얼마나 볼지'를 정한다. <sep> 위치는 SQL 첫 토큰을",
         "예측하는 자리라, 여기서 많이 보는 토큰이 SQL 을 만들 때 참고하는 질문 부분이다. 헤드·층마다 보는 곳이 다르다",
         "(--layer 로 다른 층 보기). <bos> 에 많이 몰리는 헤드는 '볼 곳이 없을 때 쉬는 자리'로 쓰는 경우가 많다.")
    x1 = x + attn.out_proj((w @ v.transpose(1, 2)).transpose(1, 2).reshape(1, T, -1))
    print(f"    5) out_proj 후 residual 더하기: {stats(x1)}")
    h2 = blk.ffn_norm(x1)
    gate, up = blk.ffn.w1(h2), blk.ffn.w3(h2)
    print(f"    6) ffn_norm → SwiGLU: w1(gate) {tuple(gate.shape)}, w3(up) {tuple(up.shape)}, silu(gate)*up → w2")
    note(f"SwiGLU: 256차원을 {gate.size(-1)}차원으로 넓혔다가(w1·w3) 다시 256으로 줄인다(w2). silu(gate) 가 up 의 각 차원을",
         "얼마나 통과시킬지 정하는 문 역할을 한다.")
    out = x1 + blk.ffn.w2(F.silu(gate) * up)
    print(f"    7) residual 더하기 → 블록 {li} 출력: {stats(out)}")
    same = torch.allclose(out, blk(x, freqs_cis=freqs), atol=1e-5)
    print(f"    손으로 계산한 결과가 블록 forward 결과와 같음: {same}")
    note("True 면 위 1)~7) 설명이 실제 구현(transformer.py)과 같은 계산이라는 뜻이다.")
    return out


def demo_model(args, model, tok: bpe.Tokenizer, prompt: list[int], h: torch.Tensor, sql: str | None) -> torch.Tensor:
    """마지막 블록 출력 h → 최종 RMSNorm → 출력층(= 임베딩 가중치 공유) → logits. logits 를 반환한다."""
    header(f"전체 모델의 출력층  (model.py)  {args.model}")
    note("마지막 블록 출력 → 최종 RMSNorm → 출력층 → 위치마다 vocab 개수만큼의 점수(logits). 출력층은 임베딩 표를",
         "그대로 다시 쓴다(weight tying): '이 벡터가 어느 토큰의 임베딩과 가장 비슷한가'로 다음 토큰 점수를 낸다.")
    counts = {"tok_emb (= lm_head, 가중치 공유)": sum(p.numel() for p in model.tok_emb.parameters()),
              f"blocks × {len(model.blocks)}": sum(p.numel() for p in model.blocks.parameters()),
              "final norm": sum(p.numel() for p in model.norm.parameters())}
    for k, v in counts.items():
        print(f"  {k:<32} {v:>10,}")
    print(f"  {'합계':<32} {model.num_params():>10,}")
    note("임베딩과 출력층이 같은 가중치라 한 번만 센다. 파라미터의 대부분은 디코더 블록에 있다.")
    section(f"입력: 마지막 블록 출력 {tuple(h.shape)}")
    with torch.no_grad():
        logits = model.lm_head(model.norm(h))
        same = torch.allclose(logits, model(torch.tensor([prompt])), atol=1e-4)
    print(f"  출력 logits {tuple(logits.shape)} (위치마다 vocab {logits.size(-1)}개 점수). "
          f"model(ids) 를 한 번에 부른 결과와 같음: {same}")
    note("True 면 앞 단계들이 넘긴 값으로 계산한 결과가 모델 전체를 한 번에 돌린 결과와 같다.")
    probs = logits[0, -1].softmax(-1)
    print("  <sep> 다음 토큰(= SQL 첫 토큰) 확률 상위 5개:")
    for p, i in zip(*probs.topk(5)):
        print(f"    {i.item():>5}  {token_str(tok, i.item()):<14} {p.item():.2e}")
    note("softmax 로 점수를 확률로 바꾼 것이다. 1위가 1에 가까우면 모델이 확신하는 것이다 (SQL 은 항상 SELECT 로 시작).")
    if sql:
        question = tok.decode(prompt[1:-1])
        b = collate_fn([tokenize_pairs([{"question": question, "sql": sql}], tok)[0]])
        with torch.no_grad():
            lg = model(b["input_ids"])
        m = b["loss_mask"][0]
        loss = F.cross_entropy(lg[0][m], b["target_ids"][0][m])
        section(f"teacher forcing loss (학습 때 계산하는 값): 정답 SQL {sql!r}")
        print(f"  SQL 토큰 {int(m.sum())}개 평균 cross entropy = {loss.item():.2e} "
              f"(정답 토큰 평균 확률 {math.exp(-loss.item()):.6f})")
        found = find_sql_entry(sql)
        split = found[0] if found else None
        note("정답 토큰을 하나씩 넣어 주며(teacher forcing) 각 위치에서 정답 다음 토큰에 준 확률의 -log 평균이다.",
             "0 에 가까울수록 정답을 확신한다. " + ({"train": "학습에 있던 SQL 이라 낮은 게 정상이다.",
                                                     None: "직접 쓴 SQL 이다."}.get(split, "학습에 없던 SQL 인데도 낮으면 일반화가 된 것이다.")))
    return logits


def demo_generate(args, model, tok: bpe.Tokenizer, prompt: list[int], gold: str | None = None) -> str | None:
    """greedy 생성을 한 토큰씩 보여 주고 생성 SQL 을 반환한다. gold 가 있으면 비교하고 DB 에서 둘 다 실행한다."""
    header("생성  (model.generate 와 같은 greedy, 한 토큰씩)")
    note("매 스텝 지금까지의 토큰열 전체를 모델에 넣어 다음 토큰 확률을 구하고, 가장 높은 토큰 하나를 붙인다(greedy).",
         f"<eos> 가 나오면 멈춘다. 2위 후보 확률이 {CONFUSED_P:.0%} 이상인 스텝은 ← 로 표시한다 (모델이 헷갈린 곳).")
    print(f"입력 프롬프트: {[token_str(tok, i) for i in prompt]}  ({len(prompt)}토큰)")
    out: list[int] = []
    confused: list[str] = []
    with torch.no_grad():
        for step in range(64):
            if len(prompt) + len(out) >= model.max_seq_len:
                break
            probs = model(torch.tensor([prompt + out]))[0, -1].softmax(-1)
            top_p, top_i = probs.topk(2)
            mark = ""
            if top_p[1].item() >= CONFUSED_P:
                mark = "  ← 헷갈림"
                confused.append(token_str(tok, top_i[0].item()))
            print(f"  step {step:>2}: {token_str(tok, top_i[0].item()):<16} p={top_p[0].item():.6f} "
                  f"(2위 {token_str(tok, top_i[1].item())} {top_p[1].item():.2e}){mark}")
            if top_i[0].item() == EOS_ID:
                break
            out.append(top_i[0].item())
    pred = safe_decode(out, tok)
    section(f"출력 SQL: {pred}")
    if confused:
        note(f"헷갈린 스텝 {len(confused)}개: {', '.join(confused)}. 이름·숫자 값은 외운 목록이 아니라 질문에서 복사해야 해서",
             "확률이 상대적으로 낮게 나오기 쉽다.")
    else:
        note("모든 스텝에서 1위 확률이 압도적이다 - 모델이 이 질문의 SQL 을 확신한다.")
    con = sqlite3.connect(args.db)
    for label, s in (("생성", pred), ("정답", gold)):
        if s is None:
            continue
        try:
            res = con.execute(s).fetchall()
            print(f"  {label} SQL 실행: {len(res)}행 {res[:3]}")
        except sqlite3.Error as e:
            print(f"  {label} SQL 실행 오류: {e}")
    con.close()
    if gold is not None:
        print(f"  정답 SQL 과 같음(EM): {pred == gold}")
    if pred is None:
        note("디코딩할 수 없는 토큰열을 만들었다 (깨진 바이트 등). 채점에서는 항상 오답이다.")
    return pred


def demo_score(args, question: str, gold: str, pred: str | None, stage: int = config.STAGE) -> dict:
    """생성 SQL 을 정답과 채점한다 (evaluation/scoring.py 가 평가셋 문항마다 하는 것과 같은 판정). 판정 결과를 반환한다."""
    rules = load_rules(stage)
    header(f"채점  (evaluation/scoring.py + {stage}단계 규칙 src/data/stage{stage}/sql_rules.py)")
    note("평가셋의 문항 하나를 채점하는 방식이다. 주 지표는 EM(생성 SQL 이 정답 문자열과 완전히 같은가)이고,",
         "나머지는 오답을 이해하기 위한 보조 지표다. 평가셋 전체 점수는 이 기록들의 비율이다.")
    section("입력")
    print(f"  질문     : {question!r}")
    print(f"  정답 SQL : {gold}")
    print(f"  생성 SQL : {pred}")
    con = sqlite3.connect(args.db)
    gold_res, pred_res = execute(con, gold), execute(con, pred)
    con.close()
    found = find_sql_entry(gold)
    r = {
        "em": pred == gold,
        "select_ambiguous": rules.select_ambiguous(question, gold),
        "em_multi": rules.multi_match(question, gold, pred),
        "exec_match": pred == gold or (gold_res is not None and pred_res == gold_res),
        "label_mismatch": rules.label_mismatch(question, gold),
        "error_parts": [] if pred == gold else rules.error_parts(gold, pred),
    }
    section("출력 (평가셋 문항 하나의 기록)")
    print(f"  EM (문자열 완전 일치)           : {r['em']}")
    print(f"  다중 정답 인정 문항             : {r['select_ambiguous']}  ({rules.MULTI_NOTE})")
    print(f"  다중 정답 EM                    : {r['em_multi']}")
    print(f"  실행 정확도 (DB 실행 결과 비교) : {r['exec_match']}  (정답 {sum(gold_res.values()) if gold_res else gold_res}행, "
          f"생성 {sum(pred_res.values()) if pred_res is not None else '실행 오류'}행)")
    print(f"  라벨 불일치 (규칙 일치 EM 제외) : {r['label_mismatch']}")
    print(f"  틀린 부분                       : {r['error_parts'] or '없음'}")
    if found:
        split, e = found
        print(f"  집계 위치: 분할·계층 {e.get('holdout_tier', split)}, WHERE 컬럼 {e['table']}.{e['where_col']}")
    # 해석
    if r["em"]:
        note("완전히 맞혔다. 다른 지표도 모두 맞음으로 센다.")
    else:
        note("틀린 부분: " + "; ".join(f"{p} - {ERROR_PART_MEANING.get(p, p)}" for p in r["error_parts"]) + ".")
        if r["em_multi"]:
            note("다만 질문에 * 와 식별 컬럼 중 무엇을 원하는지 단서가 없어, SELECT 만 다른 이 답도 다중 정답 EM 에서는 정답이다.")
        if r["exec_match"]:
            note("SQL 문자열은 다르지만 DB 실행 결과가 같다 (예: 둘 다 0행). 틀린 SQL 도 우연히 같은 결과를 낼 수 있어",
                 "실행 정확도는 보조 지표로만 본다.")
    if r["label_mismatch"]:
        note("이 문항은 정답 라벨이 라벨 규칙과 달라, 규칙대로 답해도 오답이 된다. 그래서 규칙 일치 EM 에서는 뺀다.")
    if found:
        note("평가 리포트는 이 문항을 위 계층과 WHERE 컬럼 묶음에 넣어 따로 집계한다.")
    return r


# ---------------------------------------------------------------------------
# 하위 명령
# ---------------------------------------------------------------------------

def cmd_db(args):
    demo_db(args, args.sql)


def cmd_sql(args):
    demo_sql(args, args.sql)


def cmd_questions(args):
    demo_questions(args, args.sql)


def cmd_clean(args):
    if bool(args.text) != bool(args.sql):
        raise SystemExit("clean: --text 와 --sql 은 함께 준다 (둘 다 없으면 학습 쌍 전체 요약)")
    demo_clean(args, args.text, args.sql)


def cmd_name_swap(args):
    if bool(args.text) != bool(args.sql):
        raise SystemExit("name-swap: --text 와 --sql 은 함께 준다 (둘 다 없으면 학습 쌍 전체에 적용)")
    demo_name_swap(args, args.text, args.sql)


def cmd_tokenizer(args):
    demo_tokenizer(args, load_tokenizer(args), [text_of(args)] + ([args.sql] if args.sql else []))


def cmd_dataset(args):
    demo_dataset(args, load_tokenizer(args), text_of(args), sql_of(args))


def cmd_embedding(args):
    model, tok = load_model_and_tokenizer(args)
    demo_embedding(args, model, tok, prompt_of(tok, text_of(args)))


def cmd_block(args):
    model, tok = load_model_and_tokenizer(args)
    prompt = prompt_of(tok, text_of(args))
    with torch.no_grad():
        x = model.tok_emb(torch.tensor([prompt]))
    demo_block(args, model, tok, prompt, x)


def cmd_model(args):
    model, tok = load_model_and_tokenizer(args)
    prompt = prompt_of(tok, text_of(args))
    freqs = model.freqs_cis[:len(prompt)]
    with torch.no_grad():
        h = model.tok_emb(torch.tensor([prompt]))
        for blk in model.blocks:
            h = blk(h, freqs_cis=freqs)
    demo_model(args, model, tok, prompt, h, args.sql)


def cmd_generate(args):
    model, tok = load_model_and_tokenizer(args)
    demo_generate(args, model, tok, prompt_of(tok, text_of(args)), args.sql)


def cmd_score(args):
    question, gold = text_of(args), sql_of(args)
    pred = args.pred
    if pred is None:
        model, tok = load_model_and_tokenizer(args)
        pred = predict_questions(model, [question], tok, torch.device("cpu"), None)[0]
    demo_score(args, question, gold, pred)


def cmd_all(args):
    """예시 하나를 끝까지 따라간다. 각 단계의 반환값이 다음 단계의 입력이 된다."""
    sql = sql_of(args)
    note("예시 SQL 하나를 데이터 생성부터 모델 출력, 채점까지 따라간다. 각 단계의 결과가 다음 단계의 입력이 된다:",
         "SQL → DB 행 → 생성 정보 → LLM 질문 → 라벨 판정 → 이름 교체 → 학습 입력 → 토큰 → 임베딩 → 블록 → 출력층",
         "→ 생성 SQL → 채점")
    demo_db(args, sql)                                           # SQL → 읽는 행
    entry = demo_sql(args, sql)                                  # SQL → 생성 정보
    questions = demo_questions(args, sql, entry)                 # SQL·생성 정보 → LLM 질문들
    question = args.text or (questions[0] if questions else DEFAULT_TEXT)
    print(f"\n>>> 이후 단계는 질문 {question!r} 로 진행"
          + ("" if args.text else " (위 질문 중 첫 번째. 바꾸려면 --text)"))
    demo_clean(args, question, sql)                              # (질문, SQL) → 라벨 판정
    demo_name_swap(args, question, sql)                          # (질문, SQL) → 이름 바꾼 쌍 (학습용, 추론은 원래 질문)
    model, tok = load_model_and_tokenizer(args)
    demo_dataset(args, tok, question, sql, corpus_stats=False)   # (질문, SQL) → 학습 입력
    print("\n>>> 여기까지가 학습 데이터를 만드는 과정이고, 아래부터는 학습된 모델이 질문을 SQL 로 바꾸는 과정이다.")
    prompt = demo_tokenizer(args, tok, [question])               # 질문 → <bos> 질문 <sep>
    x = demo_embedding(args, model, tok, prompt)                 # 토큰 ID → 임베딩
    h = demo_block(args, model, tok, prompt, x)                  # 임베딩 → 마지막 블록 출력
    demo_model(args, model, tok, prompt, h, sql)                 # 블록 출력 → logits, 정답 loss
    pred = demo_generate(args, model, tok, prompt, sql)          # 프롬프트 → 생성 SQL → DB 실행
    demo_score(args, question, sql, pred)                        # (질문, 정답, 생성 SQL) → 채점


# (하위 명령, 함수, 설명, 인자 묶음, --text/--sql 을 주지 않았을 때 쓰는 예시)
COMMANDS = [
    ("db", cmd_db, "DB 테이블·행", ("db", "sql"),
     "--sql 이 읽는 테이블과 조건에 맞는 행. 없으면 테이블마다 앞쪽 --n 행과 미등장 값"),
    ("sql", cmd_sql, "SQL 생성 정보", ("db", "sql", "sql_file"),
     "--sql 을 SQL 파일들(학습·분포 내·미등장 값)에서 찾은 생성 정보와 실행 결과. 없으면 --sql-file 에서 무작위 --n 개"),
    ("questions", cmd_questions, "SQL → 질문 → 검증", ("sql", "sql_file", "pairs"),
     "--sql 에 LLM 이 쓴 질문 전부(학습·평가 쌍에서 찾음). 없으면 질문이 있는 SQL 무작위 --n 개"),
    ("clean", cmd_clean, "라벨 규칙 판정", ("text", "sql", "train"),
     "--text·--sql 쌍의 판정. 없으면 학습 쌍 전체의 제거 대상 수와 SELECT 모호 문항 무작위 --n 개"),
    ("name-swap", cmd_name_swap, "이름 교체", ("text", "sql", "train", "swap"),
     "--text·--sql 쌍을 가짜 이름으로 바꾼 결과. 없으면 학습 쌍 전체에 --ratio 로 적용한 통계와 바뀐 쌍 --n 개"),
    ("tokenizer", cmd_tokenizer, "텍스트 → 토큰 → 텍스트", ("text", "sql", "tok"),
     f"--text (없으면 {DEFAULT_TEXT!r}). --sql 을 주면 SQL 도 토큰화"),
    ("dataset", cmd_dataset, "(질문, SQL) → 학습 입력", ("text", "sql", "train", "tok"),
     f"--text·--sql (없으면 {DEFAULT_TEXT!r}, {DEFAULT_SQL!r}). 전체 길이 통계는 --train-files 와 분포 내 평가셋"),
    ("embedding", cmd_embedding, "임베딩·RoPE", ("text", "model"), f"--text (없으면 {DEFAULT_TEXT!r})"),
    ("block", cmd_block, "디코더 블록 내부", ("text", "model"),
     f"--text (없으면 {DEFAULT_TEXT!r}). --layer 블록을 자세히, 나머지는 출력 통계만"),
    ("model", cmd_model, "출력층·다음 토큰 확률·loss", ("text", "sql", "model"),
     f"--text (없으면 {DEFAULT_TEXT!r}). --sql 을 주면 그 정답의 loss"),
    ("generate", cmd_generate, "토큰 단위 greedy 생성", ("db", "text", "sql", "model"),
     f"--text (없으면 {DEFAULT_TEXT!r}). --sql 을 주면 생성 결과와 비교·실행"),
    ("score", cmd_score, "생성 SQL 채점", ("db", "text", "sql", "pred", "model"),
     f"--text·--sql (없으면 {DEFAULT_TEXT!r}, {DEFAULT_SQL!r}). 생성 SQL 은 --pred, 없으면 모델이 생성"),
    ("all", cmd_all, "예시 하나로 전 단계 연결", ("db", "text", "sql", "sql_file", "pairs", "train", "swap", "model"),
     f"--sql (없으면 {DEFAULT_SQL!r}) 을 따라가고, 질문은 --text 또는 그 SQL 에 LLM 이 쓴 첫 질문"),
]


def main(argv: list[str] | None = None) -> int:
    global QUIET
    ap = argparse.ArgumentParser(
        description="파이프라인 단계별 입력 → 출력 보기", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--text/--sql 을 주지 않을 때 쓰는 예시:\n"
               + "\n".join(f"  {name:<10} {basis}" for name, _, _, _, basis in COMMANDS))
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn, help_, groups, basis in COMMANDS:
        p = sub.add_parser(name, help=help_, description=f"{help_}. 예시: {basis}")
        p.set_defaults(fn=fn)
        p.add_argument("--n", type=int, default=3, help="예시 수 (--text/--sql 을 주지 않았을 때)")
        p.add_argument("--seed", type=int, default=0, help="무작위 예시를 고르는 seed")
        p.add_argument("--quiet", action="store_true", help="설명(※) 줄을 끄고 값만 출력")
        if "text" in groups:
            p.add_argument("--text", default=None, help="질문")
        if "sql" in groups:
            p.add_argument("--sql", default=None, help="SQL")
        if "pred" in groups:
            p.add_argument("--pred", default=None, help="채점할 생성 SQL (없으면 --model 로 생성)")
        if "db" in groups:
            p.add_argument("--db", default=str(config.DB_PATH), help=f"DB (기본 {config.DB_PATH})")
        if "db" in groups or "swap" in groups:
            p.add_argument("--holdout", default=str(config.HOLDOUT_PATH), help=f"미등장 값 (기본 {config.HOLDOUT_PATH})")
        if "sql_file" in groups:
            p.add_argument("--sql-file", default=str(config.SQL_TRAIN_PATH),
                           help=f"무작위 예시를 고를 SQL 파일 (기본 {config.SQL_TRAIN_PATH})")
        if "pairs" in groups:
            p.add_argument("--pairs", nargs="+", default=None,
                           help=f"질문을 찾을 쌍 파일들 (기본: {config.TRAIN_DIR}/*.json + 평가셋 pairs.json 전부)")
        if "train" in groups or "pairs" in groups:
            p.add_argument("--eval-dir", default=str(config.EVAL_DIR), help=f"평가셋 폴더 (기본 {config.EVAL_DIR})")
        if "train" in groups:
            p.add_argument("--train-files", nargs="+", default=None, help=f"학습 쌍 파일 (기본 {config.TRAIN_DIR}/*.json)")
        if "swap" in groups:
            p.add_argument("--ratio", type=float, default=0.4, help="이름 교체 비율 (학습 쌍 전체에 적용할 때)")
            p.add_argument("--wordlist", default=str(config.WORDLIST_PATH), help="영단어 목록")
        if "tok" in groups:
            p.add_argument("--tokenizer", default=None, help=f"토크나이저 (기본 {config.TOKENIZER_PATH})")
        if "model" in groups:
            p.add_argument("--model", default=str(config.MODEL_PATH), help=f"모델 (기본 {config.MODEL_PATH})")
            p.add_argument("--tokenizer", default=None, help="토크나이저 (기본: 모델 폴더의 tokenizer.json)")
            p.add_argument("--layer", type=int, default=0, help="자세히 볼 디코더 블록 번호")
    args = ap.parse_args(argv)
    QUIET = args.quiet
    torch.manual_seed(args.seed)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
