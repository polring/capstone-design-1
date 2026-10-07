"""
predict.py — 학습된 체크포인트로 질문을 SQL 로 바꾸고 shop.db 에서 실행해 본다

저장소 루트에서 실행한다 (경로는 src/config.py, cwd 기준 상대 경로).

    python -m src.predict                                   # 대화형: 질문을 한 줄씩 입력
    python -m src.predict -q "what city does ashley live in?"
    python -m src.predict --ckpt runs/stage1/ffn683/best.pt --no-exec

- 기본 체크포인트는 `python -m src.train` 기본 설정이 저장하는 곳(runs/stage<N>/ffn<D>_swap40/best.pt,
  1단계는 runs/stage1/ffn683_swap40/best.pt)이다.
- 질문은 학습 데이터처럼 소문자로 바꿔 넣는다. 생성은 greedy 이며 제약 디코더는 없다.
- 1단계 범위(단일 테이블, SELECT 컬럼 하나 또는 *, WHERE 등호 하나) 밖의 질문에도 무언가를 출력하지만
  의미는 없다.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import torch

from src import config
from src.data.dataset import BOS_ID, EOS_ID, SEP_ID
from src.evaluation import load_model, load_rules, safe_decode, tokenizer_path_for
from src.tokenizer import bpe

# train.py 기본 실행(이름 교체 40%)의 실행 이름과 같은 규칙
DEFAULT_CKPT = str(config.RUNS_DIR / f"ffn{config.FFN_DIM}_swap40" / "best.pt")
MAX_ROWS = 20


def predict(model, question: str, tok: bpe.Tokenizer, amp_dtype) -> str | None:
    prompt = [BOS_ID] + tok.encode(question.strip().lower()) + [SEP_ID]
    if len(prompt) >= model.max_seq_len:
        raise ValueError(f"질문이 너무 깁니다 ({len(prompt)}토큰, 최대 {model.max_seq_len - 1})")
    device = model.freqs_cis.device
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model.generate(prompt, eos_id=EOS_ID)
    return safe_decode(out, tok)


def run_sql(con: sqlite3.Connection, sql: str) -> None:
    try:
        cur = con.execute(sql)
    except sqlite3.Error as e:
        print(f"  실행 오류: {e}")
        return
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"  결과 {len(rows)}행  ({', '.join(cols)})")
    for row in rows[:MAX_ROWS]:
        print("   ", " | ".join(str(v) for v in row))
    if len(rows) > MAX_ROWS:
        print(f"    ... {len(rows) - MAX_ROWS}행 더 있음")


def answer(question: str, model, tok: bpe.Tokenizer, amp_dtype, con, rules=None) -> None:
    """질문 하나를 SQL 로 바꿔 출력하고, con 이 있으면 실행 결과도 출력한다.
    rules: 단계별 SQL 규칙 모듈 — 생성 SQL 이 그 단계 형식이 아니면 표시한다 (기본: config.STAGE)."""
    rules = rules or load_rules()
    try:
        sql = predict(model, question, tok, amp_dtype)
    except ValueError as e:
        print(f"  {e}")
        return
    if sql is None:
        print("  SQL: (디코딩 불가 — 깨진 바이트나 토크나이저에 없는 토큰 생성)")
        return
    note = "" if rules.parse_sql(sql) else f"  [{rules.STAGE}단계 SQL 형식 아님]"
    print(f"  SQL: {sql}{note}")
    if con is not None:
        run_sql(con, sql)


def main() -> int:
    ap = argparse.ArgumentParser(description="질문 → SQL 생성 및 실행")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help=f"체크포인트 (기본 {DEFAULT_CKPT})")
    ap.add_argument("-q", "--question", action="append", help="질문 (여러 번 지정 가능). 없으면 대화형")
    ap.add_argument("--no-exec", action="store_true", help="SQL 을 shop.db 에서 실행하지 않음")
    ap.add_argument("--cpu", action="store_true", help="GPU 가 있어도 CPU 사용")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        print(f"체크포인트가 없습니다: {args.ckpt}\n"
              "runs/ 는 커밋되지 않으므로 직접 학습해야 합니다 (docs/SETUP.md 7절).", file=sys.stderr)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None  # evaluation.py 와 같은 수치 조건
    model, ckpt = load_model(args.ckpt, device)
    tok_path = tokenizer_path_for(args.ckpt, ckpt)
    if not Path(tok_path).exists():
        print(f"{tok_path} 가 없습니다. 먼저 `python -m src.tokenizer.bpe` 를 실행하세요.", file=sys.stderr)
        return 1
    tok = bpe.Tokenizer.load(tok_path)
    rules = load_rules(ckpt.get("stage", config.STAGE))
    con = None if args.no_exec else sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)

    print(f"checkpoint {args.ckpt} (epoch {ckpt.get('epoch')}, val EM {ckpt.get('val_em', 0):.4f}, "
          f"{device.type})")
    if args.question:
        for q in args.question:
            print(f"Q: {q}")
            answer(q, model, tok, amp_dtype, con, rules)
        return 0

    print("질문을 입력하세요 (빈 줄 또는 Ctrl+C 로 종료)")
    while True:
        try:
            q = input("Q: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q.strip():
            break
        answer(q, model, tok, amp_dtype, con, rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
