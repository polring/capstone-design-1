"""
Text-to-SQL 로컬 LLM(Ollama) 질문 생성 파이프라인 -- 완전 자동화판 (v2).

question_gen.py(v1)는 "사람이 프롬프트를 챗봇에 붙여넣고 응답을 받아와 저장한다"는 수동 호출을
전제로 한다. 이 파일은 그 자리에 로컬 Ollama 서버 API 호출을 넣어서, 사람 개입이나 다른 LLM
도움 없이 샘플링 -> 프롬프트 생성 -> 모델 호출 -> JSON 복구 -> 규칙 검증 -> 다양성 체크까지
한 번에 끝내고 결과만 내놓는다.

Qwen3-14B처럼 JSON 형식이 자주 깨지는 모델을 위해 3단계 복구를 쓴다:
  1) json.JSONDecoder.raw_decode -- 배열이 끝난 뒤 후행 쓰레기 문자가 있어도 앞부분만 파싱
  2) 흔한 괄호 패턴 보정 -- "questions" 배열을 닫는 ']' 누락, 객체 사이 ',{' 누락을 정규식으로 보정
  3) id/questions 필드 직접 정규식 추출 -- 괄호 구조가 아예 깨져도 필드명 패턴만으로 복구
그래도 실패하면 예산이 허락하는 한 같은 배치를 재시도한다.

샘플링(stratified_sample)·프롬프트 생성(make_prompts_small_model)·규칙 검증(validate/
template_check)은 question_gen.py 걸 그대로 재사용한다 -- 검증 기준이 이 파일과 question_gen.py
사이에서 서로 다르게 갈라지는 걸 막기 위해 로직을 한 곳(question_gen.py)에만 둔다.

사용법:
    python src/data/generator/question_gen_auto.py --model qwen3:14b --n 128 --batch-size 8 \
        --questions-per-sql 8 --seed 0

    (ollama가 로컬 11434 포트에서 떠 있어야 하고, --model로 지정한 모델이 이미 받아져 있어야 함)

결과는 기본적으로 data_raw/auto_pilot/ 밑에 쌓인다(.gitignore로 커밋 안 되는 실험 영역 --
data_raw/는 원래 "검증 전 원본 라운드"를 두는 자리라는 기존 관례를 따름). 결과를 정식
학습 코퍼스로 채택하려면, 검토 후 data/pilot_<이름>_merged/pilot_train_pairs.json 같은
곳으로 사람이 직접 옮긴다(이번 Qwen3-14B 파일럿에서 pilot_qwen3_merged/를 만든 것과 동일한
방식) -- 이 스크립트가 자동으로 정식 코퍼스에 병합하지는 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

try:
    from . import (
        question_gen as qg,
    )  # 패키지로 import될 때 (src.data.generator.question_gen_auto)
except ImportError:
    import question_gen as qg  # 스크립트로 직접 실행될 때

OLLAMA_API = "http://127.0.0.1:11434/api/generate"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

QSTR = r'"(?:[^"\\]|\\.)*"'
_ITEM_RE = re.compile(
    r'"id"\s*:\s*(\d+)\s*,\s*"questions"\s*:\s*\[?\s*'
    r"((?:" + QSTR + r"\s*,\s*)*" + QSTR + r")"
)


# ---------------------------------------------------------------------------
# 1. JSON 추출 및 복구 (3단계)
# ---------------------------------------------------------------------------


def _try_parse(text: str) -> list | None:
    try:
        data, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _expected_item_count(text: str) -> int:
    """괄호가 깨져도 '"id":' 개수는 보존되므로, 이걸로 원래 몇 개 항목이었는지 추정."""
    return len(re.findall(r'"id"\s*:', text))


def _repair_bracket_patterns(text: str) -> list | None:
    """관찰된 반복 오류 두 가지를 보정한다:
    1) "questions" 배열을 닫는 ']'를 빠뜨리고 바로 객체를 닫아버림.
    2) 객체 사이 콤마를 빠뜨림 ('}{' 형태).
    복구 결과 항목 개수가 원래(_expected_item_count)와 다르면 일부만 조용히 잘려나간
    거짓 성공일 수 있으므로 버린다."""
    expected_n = _expected_item_count(text)

    fix = re.sub(r'"\}', '"]}', text)
    fix = re.sub(r"\}\{", "},{", fix)
    data = _try_parse(fix)
    if data is not None and len(data) == expected_n:
        return data

    positions = [m.start() + 1 for m in re.finditer(r'"\}', text)]
    for pos in reversed(positions):
        candidate = text[:pos] + "]" + text[pos:]
        data = _try_parse(candidate)
        if data is not None and len(data) == expected_n:
            return data
    return None


def _regex_extract_items(text: str) -> list | None:
    """괄호 구조가 어떻게 꼬였든(순서가 뒤바뀌거나 '{'/'}' 가 통째로 빠지는 등) 상관없이
    "id":<숫자>,"questions":[<따옴표 문자열들>] 패턴만 직접 뽑아낸다."""
    items = []
    for m in _ITEM_RE.finditer(text):
        qid = int(m.group(1))
        questions = [json.loads(q) for q in re.findall(QSTR, m.group(2))]
        items.append({"id": qid, "questions": questions})
    if not items or len(items) != _expected_item_count(text):
        return None
    return items


def extract_json_array(raw: str) -> list | None:
    text = THINK_RE.sub("", raw).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("[")
    if start == -1:
        return None
    body = text[start:]
    data = _try_parse(body)
    if data is not None:
        return data
    data = _repair_bracket_patterns(body)
    if data is not None:
        return data
    return _regex_extract_items(body)


# ---------------------------------------------------------------------------
# 2. Ollama 호출
# ---------------------------------------------------------------------------


def call_ollama(
    model: str, prompt: str, timeout: int
) -> tuple[str, list | None, float]:
    """ollama REST API를 직접 호출한다 (CLI는 스트리밍 렌더링용 ANSI 코드가 표준출력
    캡처에 섞여 들어와 텍스트가 깨지는 문제가 있어서 API를 씀). think=false로 Qwen3
    계열의 기본 사고 모드를 끈다."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "think": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_API, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        raw = data.get("response", "")
    except Exception as e:
        raw = f"__REQUEST_ERROR__ {e}"
    elapsed = time.monotonic() - t0
    return raw, extract_json_array(raw), elapsed


# ---------------------------------------------------------------------------
# 3. 파이프라인 실행
# ---------------------------------------------------------------------------


def run_pipeline(
    model: str,
    train_path: str,
    n: int,
    batch_size: int,
    n_questions: int,
    seed: int,
    out_dir: str,
    exclude_dirs: list[str] | None,
    budget_seconds: int,
    per_batch_timeout: int,
    max_retries: int,
    max_consecutive_failures: int = 5,
) -> dict:
    out_path = Path(out_dir)
    qg.make_prompts_small_model(
        str(out_path),
        train_path,
        n,
        batch_size,
        seed,
        exclude_dirs,
        n_questions=n_questions,
    )

    prompts_dir = out_path / "prompts"
    responses_dir = out_path / "responses"
    batch_files = sorted(prompts_dir.glob("batch_*.txt"))

    start = time.monotonic()
    done = failed = skipped = 0
    consecutive_failures = 0
    timings: list[float] = []
    aborted_early = False

    for pf in batch_files:
        resp_path = responses_dir / (pf.stem + ".json")
        if resp_path.exists():
            # 무인 실행 중 중단됐다가 같은 명령으로 재시작한 경우 -- 이미 끝난 배치는
            # API를 다시 호출하지 않고 건너뛴다 (몇 시간짜리 실행에서 중요한 안전장치).
            print(f"[{pf.name}] 이미 완료됨 (건너뜀)")
            done += 1
            skipped += 1
            continue

        if (time.monotonic() - start) >= budget_seconds - 10:
            print(f"[예산 소진] {pf.name}부터 남은 배치는 건너뜀")
            failed += len(batch_files) - (done + failed)
            break

        prompt = pf.read_text(encoding="utf-8")
        success = False
        for attempt in range(1, max_retries + 2):
            remaining = budget_seconds - (time.monotonic() - start)
            if remaining <= 5:
                break
            raw, parsed, elapsed = call_ollama(
                model, prompt, timeout=min(per_batch_timeout, int(remaining))
            )
            timings.append(elapsed)
            tag = f"[{pf.name}] 시도{attempt}: {elapsed:.1f}s"
            if parsed is not None:
                with open(resp_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                print(f"{tag} 성공 ({len(parsed)}개 항목)")
                success = True
                break
            print(
                f"{tag} 실패 ({raw[:80]!r})"
                + (" -- 재시도" if attempt <= max_retries else " -- 포기")
            )

        if success:
            done += 1
            consecutive_failures = 0
        else:
            failed += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                # 배치 몇 개가 재시도까지 다 실패했다는 건 개별 프롬프트 문제가 아니라
                # ollama 서버가 죽었거나 응답을 안 하는 상황일 가능성이 높다. 이걸 못 잡으면
                # 무인 실행 중 남은 시간 예산을 전부 실패한 호출로 허비하게 된다.
                print(
                    f"\n[중단] 배치 {consecutive_failures}개 연속 완전 실패 -- ollama 서버 상태를 "
                    f"확인하세요 (`ollama ps`). 남은 배치는 건너뜁니다."
                )
                aborted_early = True
                skipped_remaining = len(batch_files) - (batch_files.index(pf) + 1)
                failed += skipped_remaining
                break

    total_elapsed = time.monotonic() - start

    report = qg.validate(str(out_path), expected_questions=n_questions)
    tmpl_report = qg.template_check(str(out_path))

    summary = {
        "model": model,
        "n_batches": len(batch_files),
        "n_batches_done": done,
        "n_batches_skipped_already_complete": skipped,
        "n_batches_failed": failed,
        "aborted_early_server_down": aborted_early,
        "total_elapsed_sec": round(total_elapsed, 1),
        "avg_batch_sec": round(sum(timings) / len(timings), 1) if timings else None,
        "n_questions_seen": report["n_questions_seen"],
        "n_passed": report["n_passed"],
        "pass_rate": report["pass_rate"],
        "n_soft_flags": report["n_soft_flags"],
        "template_repeat_ratio": tmpl_report["overall_repeat_ratio"],
        "out_dir": str(out_path),
    }
    with open(out_path / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(
        description="로컬 LLM(Ollama) 질문 생성 파이프라인 (완전 자동화)"
    )
    ap.add_argument(
        "--model", default="qwen3:14b", help="ollama에 받아져 있는 모델 이름"
    )
    ap.add_argument("--train", default="data/sql_train.json")
    ap.add_argument("--n", type=int, default=128, help="샘플링할 SQL 개수")
    ap.add_argument("--batch-size", type=int, default=8, help="호출 1회당 SQL 개수")
    ap.add_argument(
        "--questions-per-sql",
        type=int,
        default=5,
        help="SQL 하나당 요구할 질문 개수 (최대 %d)" % len(qg.QUESTION_STYLE_SLOTS),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default="data_raw/auto_pilot",
        help="결과 저장 위치. data_raw/ 밑이 기본값 -- 검토·채택 전 실험 데이터는 "
        "여기 두는 게 프로젝트 관례(.gitignore로 커밋 안 됨). 정식 채택 시에만 "
        "data/pilot_<이름>_merged/ 같은 곳으로 수동으로 옮긴다.",
    )
    ap.add_argument(
        "--exclude-dir",
        default=None,
        help="이 디렉터리들의 pilot_sql.json에 있는 SQL은 샘플링에서 제외 (쉼표 구분)",
    )
    ap.add_argument(
        "--budget-seconds", type=int, default=1800, help="전체 실행 시간 예산"
    )
    ap.add_argument(
        "--per-batch-timeout", type=int, default=180, help="호출 1회당 타임아웃"
    )
    ap.add_argument(
        "--max-retries", type=int, default=2, help="배치 하나당 최대 재시도 횟수"
    )
    ap.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=5,
        help="이 개수만큼 배치가 연속으로 완전히 실패하면 (재시도까지 다 소진) "
        "ollama 서버 다운으로 보고 남은 배치를 접고 조기 종료 -- 무인 실행 중 "
        "죽은 서버에 시간 예산을 전부 낭비하는 걸 막는 안전장치",
    )
    args = ap.parse_args()

    exclude_dirs = args.exclude_dir.split(",") if args.exclude_dir else None
    summary = run_pipeline(
        args.model,
        args.train,
        args.n,
        args.batch_size,
        args.questions_per_sql,
        args.seed,
        args.out,
        exclude_dirs,
        args.budget_seconds,
        args.per_batch_timeout,
        args.max_retries,
        args.max_consecutive_failures,
    )

    print()
    print(f"=== 파이프라인 완료 ({summary['model']}) ===")
    if summary["aborted_early_server_down"]:
        print(
            "*** ollama 서버 응답 없음으로 조기 중단됨 -- 서버 상태 확인 후 같은 명령으로 재실행하면 "
            "이미 끝난 배치는 건너뛰고 이어서 진행됩니다 ***"
        )
    avg_str = (
        f"{summary['avg_batch_sec']}s"
        if summary["avg_batch_sec"] is not None
        else "N/A (신규 호출 없음)"
    )
    print(
        f"배치: {summary['n_batches_done']}/{summary['n_batches']} 성공"
        f" (그중 이미 완료돼 건너뜀 {summary['n_batches_skipped_already_complete']}개)"
        f"{', 최종 실패 ' + str(summary['n_batches_failed']) + '개' if summary['n_batches_failed'] else ''}, "
        f"총 {summary['total_elapsed_sec']:.0f}s (호출당 평균 {avg_str})"
    )
    print(
        f"검증 통과율: {summary['pass_rate']:.1%} ({summary['n_passed']}/{summary['n_questions_seen']}), "
        f"수동확인 flag {summary['n_soft_flags']}건"
    )
    print(f"틀(skeleton) 재사용 비율: {summary['template_repeat_ratio']:.1%}")
    print(f"결과: {summary['out_dir']}/pilot_train_pairs.json")
    print(f"요약: {summary['out_dir']}/pipeline_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
