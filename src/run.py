"""
run.py — 추론·평가 통합 명령 (이슈 #26)

저장소 루트에서 실행한다. 입력 형태와 모드에 따라 동작이 정해진다.

    python -m src.run                                         # 대화형: 질문을 한 줄씩 입력
    python -m src.run --input "what city does ashley live in?"   # 단건: SQL + DB 실행 결과 출력
    python -m src.run --input questions.json --test           # JSON 배치 추론 → 예측 SQL 파일 저장
    python -m src.run --input holdout --evaluation            # 평가셋(이름·폴더·JSON) 채점 → 리포트 출력·저장
    python -m src.run --evaluation                            # 이 단계의 평가셋 전부 채점
    python -m src.run --eval-sets indist holdout              # 고른 평가셋만 채점 (--test 를 주면 추론만)
    python -m src.run --stage 2 --evaluation                  # 2단계 기본 모델·평가셋·채점 규칙으로
    python -m src.run --list-sets                             # 평가셋 목록
    python -m src.run --build-sql indist --sql data/stage1/sql/eval_indist.json   # 평가셋의 sql.json 생성

인자 (생략하면 --stage 단계의 config 기본값, --stage 생략 시 config.STAGE):
    --model     체크포인트 파일, 또는 폴더(안의 model.pt, 없으면 best.pt). 기본 models/stage<N>/model.pt
    --input     질문 문자열 / .json 파일 / 평가셋 폴더 / 평가셋 이름 (--data-set 도 같음). 생략하면 대화형
                (--evaluation 이면 평가셋 전부). JSON 은 [{"question": ..., "sql": ...}] 또는 ["질문", ...].
    --test      채점 없이 추론만. 파일 입력이면 예측을 JSON 으로 저장한다.
    --evaluation  정답과 비교해 EM·다중 정답 EM·실행 정확도·신뢰구간·계층별·오답 분류를 낸다.
                모드를 생략하면 문자열은 --test, 파일은 정답(sql)이 모두 있으면 --evaluation 이다.
                정답이 없는 입력에 --evaluation 을 주면 안내 후 --test 로 진행한다.
    --eval-sets  --input 없이 쓸 때 처리할 평가셋 이름들 (기본: 전부). 모드를 생략하면 채점한다.
    --output    결과 저장 위치 (--out 도 같음). .json 이면 그 파일(평가셋이 여러 개면 한 파일로 묶음), 아니면 폴더. 기본: 평가는 runs/ 안의 체크포인트면 그 폴더,
                배포 모델이면 runs/stage<N>/release_eval/ 의 eval_<set>.json, 추론은 runs/stage<N>/predictions/<이름>.json
    --tokenizer, --db, --eval-dir  읽을 파일 바꾸기 (기본: 모델 폴더의 tokenizer.json, 단계의 shop.db, 단계의 eval/)

- 모델 불러오기·생성은 src/inference/, 채점·평가셋은 src/evaluation/ (train.py, release.py 와 같은 함수).
- 질문은 학습 데이터처럼 소문자로 바꿔 넣는다. 생성은 greedy 이며 제약 디코더는 없다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import torch

from src import config
from src.data.dataset import BOS_ID, EOS_ID, SEP_ID
from src.evaluation.eval_sets import default_out_dir, list_sets, load_eval_set, write_sql_meta
from src.evaluation.scoring import evaluate_pairs, load_rules, print_report
from src.inference.generate import predict_questions, safe_decode
from src.inference.loading import describe_checkpoint, load_model, tokenizer_path_for
from src.tokenizer import bpe

MAX_ROWS = 20


# ---------------------------------------------------------------------------
# 단건 추론
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 입력·모델 해석
# ---------------------------------------------------------------------------

def resolve_model(model: str | None, paths: config.StagePaths) -> Path:
    """파일이면 그대로, 폴더면 안의 model.pt (없으면 best.pt), 생략하면 단계의 배포 모델."""
    if model is None:
        return paths.model_path
    p = Path(model)
    if p.is_dir():
        for name in ("model.pt", "best.pt"):
            if (p / name).is_file():
                return p / name
        raise FileNotFoundError(f"{p} 안에 model.pt / best.pt 가 없음")
    return p


def resolve_input(value: str | None, eval_dir: str | Path) -> dict:
    """--input 해석. 반환 kind:
    none: 입력 없음 / question: 질문 문자열 / pairs: 쌍 목록 (name, pairs, meta=SQL 메타데이터 또는 None)."""
    if value is None:
        return {"kind": "none"}
    p = Path(value)
    if p.is_dir():                                   # 평가셋 폴더
        pairs, meta = _load_pairs_file(p / config.PAIRS_FILE)
        return {"kind": "pairs", "name": p.name, "pairs": pairs, "meta": meta}
    if p.suffix.lower() == ".json":
        if not p.is_file():
            raise FileNotFoundError(f"입력 파일이 없음: {p}")
        pairs, meta = _load_pairs_file(p)
        name = p.parent.name if p.name == config.PAIRS_FILE else p.stem
        return {"kind": "pairs", "name": name, "pairs": pairs, "meta": meta}
    if value in list_sets(eval_dir):                 # 평가셋 이름
        pairs, meta = load_eval_set(value, eval_dir)
        return {"kind": "pairs", "name": value, "pairs": pairs, "meta": meta}
    return {"kind": "question", "question": value}


def _load_pairs_file(path: Path) -> tuple[list[dict], dict | None]:
    """[{"question", "sql"?}] 또는 ["질문", ...]. 같은 폴더에 sql.json 이 있으면 메타데이터로 함께 읽는다."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = [{"question": x} if isinstance(x, str) else x for x in data]
    if not pairs or any("question" not in x for x in pairs):
        raise ValueError(f"{path}: 질문 목록이 비었거나 question 이 없는 항목이 있음")
    meta_path = path.parent / config.EVAL_SQL_FILE
    meta = None
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = {r["sql"]: r for r in json.load(f)}
    return pairs, meta


def output_path(output: str | None, default_dir: Path, filename: str) -> Path:
    """--output 이 .json 이면 그 파일, 폴더면 그 안의 filename, 없으면 default_dir/filename."""
    if output is None:
        return default_dir / filename
    p = Path(output)
    return p if p.suffix.lower() == ".json" else p / filename


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# 모드별 실행
# ---------------------------------------------------------------------------

def run_evaluation(sets: list[dict], model, tok, device, amp_dtype, rules, args, out_default: Path) -> None:
    combined = {}
    single_file = args.output is not None and Path(args.output).suffix.lower() == ".json"
    for s in sets:
        result = evaluate_pairs(model, s["pairs"], tok, device, amp_dtype, s["name"], s["meta"], args.db, rules,
                                args.batch_size)
        print_report(result)
        for f in result["failures"][:args.show]:
            print(f"    Q: {f['question']}\n       gold: {f['gold']}\n       pred: {f['pred']}  "
                  f"({'+'.join(f['error_parts'])})")
        if single_file and len(sets) > 1:
            combined[s["name"]] = result
        else:
            save_json(output_path(args.output, out_default, f"eval_{s['name']}.json"), result)
    if combined:
        save_json(Path(args.output), combined)


def run_batch_test(sets: list[dict], model, tok, device, amp_dtype, rules, args, out_default: Path) -> None:
    combined = {}
    single_file = args.output is not None and Path(args.output).suffix.lower() == ".json"
    for s in sets:
        records = predict_records(s, model, tok, device, amp_dtype, rules, args)
        if single_file and len(sets) > 1:
            combined[s["name"]] = records
        else:
            save_json(output_path(args.output, out_default, f"{s['name']}.json"), records)
    if combined:
        save_json(Path(args.output), combined)


def predict_records(s: dict, model, tok, device, amp_dtype, rules, args) -> list[dict]:
    """쌍 목록 s 를 배치로 추론해 [{"question", "pred", "stage_format", ("sql")}] 를 만들고 요약을 출력한다."""
    questions = [p["question"] for p in s["pairs"]]
    preds = predict_questions(model, questions, tok, device, amp_dtype, args.batch_size)
    records = []
    for p, pred in zip(s["pairs"], preds):
        r = {"question": p["question"], "pred": pred,
             "stage_format": pred is not None and rules.parse_sql(pred) is not None}
        if "sql" in p:
            r["sql"] = p["sql"]
        records.append(r)
    n_none = sum(r["pred"] is None for r in records)
    n_fmt = sum(r["stage_format"] for r in records)
    print(f"== {s['name']}: 질문 {len(records)}개 추론 (배치 {args.batch_size}): {rules.STAGE}단계 SQL 형식 {n_fmt}개, "
          f"생성 실패(너무 긴 질문·디코딩 불가) {n_none}개")
    for r in records[:args.show]:
        print(f"    Q: {r['question']}\n       pred: {r['pred']}")
    return records


def interactive(model, tok, amp_dtype, con, rules) -> None:
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="질문 → SQL 추론 / 평가 (통합 명령)")
    ap.add_argument("--stage", type=int, default=None,
                    help=f"단계: 기본 모델·평가셋·DB·채점 규칙을 이 단계로 (기본 config.STAGE={config.STAGE})")
    ap.add_argument("--model", default=None, help="체크포인트 파일 또는 폴더 (기본: 단계의 models/stage<N>/model.pt)")
    ap.add_argument("--input", "--data-set", dest="input", default=None,
                    help="질문 문자열 / .json 파일 / 평가셋 폴더 / 평가셋 이름 (생략: 대화형, --evaluation 이면 평가셋 전부)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true", help="채점 없이 추론만")
    mode.add_argument("--evaluation", action="store_true", help="정답과 비교해 채점")
    ap.add_argument("--eval-sets", nargs="+", default=None,
                    help="--input 없이 쓸 때 처리할 평가셋 이름들 (기본: 전부). 모드 생략 시 채점")
    ap.add_argument("--output", "--out", dest="output", default=None, help="결과 저장 위치 (.json 파일 또는 폴더)")
    ap.add_argument("--tokenizer", default=None, help="토크나이저 (기본: 모델 폴더의 tokenizer.json, 없으면 체크포인트에 적힌 경로)")
    ap.add_argument("--db", default=None, help="SQL 실행·실행 정확도용 DB (기본: 단계의 shop.db)")
    ap.add_argument("--eval-dir", default=None, help="평가셋 상위 폴더 (기본: 단계의 eval/)")
    ap.add_argument("--batch-size", type=int, default=512, help="배치 추론·평가의 생성 배치 크기")
    ap.add_argument("--show", type=int, default=10, help="출력할 오답·예측 예시 수")
    ap.add_argument("--no-exec", action="store_true", help="단건·대화형에서 SQL 을 DB 에서 실행하지 않음")
    ap.add_argument("--cpu", action="store_true", help="GPU 가 있어도 CPU 사용")
    ap.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기")
    ap.add_argument("--list-sets", action="store_true", help="평가셋 목록과 쌍·SQL 수를 출력하고 끝낸다")
    ap.add_argument("--build-sql", metavar="NAME", help="평가셋 NAME 의 sql.json 을 --sql 파일에서 만들고 끝낸다")
    ap.add_argument("--sql", help="--build-sql 용 sql_gen 출력 파일 (예: data/stage1/sql/eval_indist.json)")
    args = ap.parse_args(argv)

    stage = args.stage or config.STAGE
    paths = config.stage_paths(stage)
    eval_dir = Path(args.eval_dir or paths.eval_dir)
    args.db = args.db or str(paths.db_path)

    # --- 평가셋 관리 (모델 불필요) ---
    if args.list_sets:
        for name in list_sets(eval_dir):
            pairs, meta = load_eval_set(name, eval_dir)
            print(f"{name:<16} 쌍 {len(pairs):>6}  SQL {len(meta):>5}")
        return 0
    if args.build_sql:
        if not args.sql:
            ap.error("--build-sql 에는 --sql 이 필요하다")
        print(f"{write_sql_meta(args.build_sql, args.sql, eval_dir)} 생성")
        return 0

    # --- 입력 해석 ---
    if args.eval_sets:
        if args.input is not None:
            ap.error("--eval-sets 는 --input 과 함께 쓰지 않는다 (평가셋 하나는 --input <이름>)")
        unknown = sorted(set(args.eval_sets) - set(list_sets(eval_dir)))
        if unknown:
            ap.error(f"{eval_dir} 에 없는 평가셋: {unknown} (있는 것: {list_sets(eval_dir)})")
    try:
        inp = resolve_input(args.input, eval_dir)
        model_path = resolve_model(args.model, paths)
    except (FileNotFoundError, ValueError) as e:
        ap.error(str(e))
    if not model_path.is_file():
        print(f"모델이 없습니다: {model_path}\n배포 모델은 python -m src.release --ckpt <best.pt> 로 만든다.",
              file=sys.stderr)
        return 1

    # --- 모델 ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None
    model, ckpt = load_model(model_path, device)
    tok_path = args.tokenizer or tokenizer_path_for(model_path, ckpt)
    if not Path(tok_path).exists():
        print(f"{tok_path} 가 없습니다. 먼저 `python -m src.tokenizer.bpe` 를 실행하세요.", file=sys.stderr)
        return 1
    tok = bpe.Tokenizer.load(tok_path)
    rules = load_rules(stage)
    print(f"checkpoint {model_path} ({describe_checkpoint(ckpt)}, {device.type}, {stage}단계 규칙)")
    if ckpt.get("stage", stage) != stage:
        print(f"  주의: 모델은 {ckpt['stage']}단계인데 {stage}단계 규칙·데이터로 실행한다")

    # --- 모드 결정 ---
    if inp["kind"] == "question":
        if args.evaluation:
            print("  질문 문자열에는 정답이 없어 --test(단건 추론)로 진행한다")
        con = None if args.no_exec else sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        print(f"Q: {inp['question']}")
        answer(inp["question"], model, tok, amp_dtype, con, rules)
        return 0

    if inp["kind"] == "none":
        if args.evaluation or args.eval_sets:
            names = args.eval_sets or list_sets(eval_dir)
            sets = [{"name": n, "pairs": p, "meta": m} for n in names for p, m in [load_eval_set(n, eval_dir)]]
            if not sets:
                print(f"{eval_dir} 에 평가셋이 없습니다.", file=sys.stderr)
                return 1
            if args.test:
                run_batch_test(sets, model, tok, device, amp_dtype, rules, args, paths.predictions_dir)
            else:
                run_evaluation(sets, model, tok, device, amp_dtype, rules, args,
                               default_out_dir(model_path, paths.release_eval_dir))
            return 0
        con = None if args.no_exec else sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        interactive(model, tok, amp_dtype, con, rules)
        return 0

    has_gold = all("sql" in p for p in inp["pairs"])
    evaluate = args.evaluation or (not args.test and has_gold)
    if evaluate and not has_gold:
        print("  정답(sql)이 없는 항목이 있어 --test(배치 추론)로 진행한다")
        evaluate = False
    if evaluate:
        run_evaluation([inp], model, tok, device, amp_dtype, rules, args,
                       default_out_dir(model_path, paths.release_eval_dir))
    else:
        run_batch_test([inp], model, tok, device, amp_dtype, rules, args, paths.predictions_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
