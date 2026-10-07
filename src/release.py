"""
release.py — 학습 체크포인트를 배포용 모델 폴더(config.MODEL_DIR, 커밋 대상)로 내보낸다

저장소 루트에서 실행한다.

    python -m src.release --ckpt runs/stage1/ffn683_swap40/best.pt            # → models/stage<N>/
    python -m src.release --ckpt runs/stage1/ffn683_swap40/best.pt --no-eval  # 평가 생략 (model_info 에 eval 없음)

산출물 (이 폴더 하나로 추론할 수 있다 — run.py 의 기본 모델, C 추론 엔진 exporter 의 입력):
    model.pt        {"format_version", "stage", "model_cfg", "model"(state_dict)}. 옵티마이저 상태·학습 인자·
                    로컬 경로는 넣지 않는다. torch.load(..., weights_only=True) 로 읽힌다.
    tokenizer.json  이 모델이 학습에 쓴 토크나이저 (특수·seed 토큰 + 병합 규칙). data/stage<N>/tokenizer/ 의
                    파일은 "다음 학습용"이라 나중에 달라질 수 있으므로 모델 폴더에 따로 둔다.
    model_info.json 출처 체크포인트, 학습 설정 요약, 검증 EM, 그리고 내보낸 파일을 다시 불러 평가한 결과

단계마다 최종 모델 하나만 커밋한다 (다시 내보내면 덮어쓴다).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src import config
from src.evaluation.eval_sets import list_sets
from src.evaluation.scoring import evaluate_set, load_rules
from src.inference.loading import load_model, tokenizer_path_for
from src.tokenizer import bpe

FORMAT_VERSION = 1
# model_info.json 에 남길 학습 인자 (경로·실행 환경에 따라 달라지는 값은 뺀다)
TRAIN_ARG_KEYS = ("name_swap_ratio", "ffn_dim", "epochs", "batch_size", "lr", "weight_decay", "warmup",
                  "val_ratio", "patience", "seed")
RUN_METRIC_KEYS = ("epochs_run", "steps", "train_time_s", "avg_tokens_per_s")


def _repo_relative(path: str | Path) -> str:
    """저장소 안 경로는 상대 경로로, 밖이면 파일 이름만 남긴다 (개인 경로가 커밋되지 않게)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.name


def release(ckpt_path: str | Path, out_dir: str | Path = config.MODEL_DIR, run_eval: bool = True,
            eval_dir: str | Path = config.EVAL_DIR, db_path: str | Path = config.DB_PATH,
            tokenizer_path: str | Path | None = None) -> dict:
    """체크포인트와 그 토크나이저를 out_dir 에 저장하고 모델 정보(model_info.json 내용)를 반환한다.
    tokenizer_path 를 주지 않으면 체크포인트가 학습에 쓴 토크나이저를 찾는다 (inference.loading.tokenizer_path_for)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tok = bpe.Tokenizer.load(tokenizer_path or tokenizer_path_for(ckpt_path, ckpt))
    model_cfg = dict(ckpt["model_cfg"])
    if tok.vocab_size > model_cfg["vocab_size"]:
        raise ValueError(f"토크나이저 vocab {tok.vocab_size} 가 모델 vocab {model_cfg['vocab_size']} 보다 큼 - 짝이 맞지 않음")
    stage = ckpt.get("stage", config.STAGE)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": FORMAT_VERSION, "stage": stage, "model_cfg": model_cfg, "model": ckpt["model"]},
               out / "model.pt")
    tok.save(out / "tokenizer.json")

    args = ckpt.get("args", {})
    info: dict = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "source_checkpoint": _repo_relative(ckpt_path),
        "model_cfg": model_cfg,
        "tokenizer": {"vocab_size": tok.vocab_size, "n_seed_tokens": len(tok.seed_tokens), "n_merges": len(tok.merges)},
        "train_args": {k: args[k] for k in TRAIN_ARG_KEYS if k in args},
        "best_epoch": ckpt.get("epoch"), "best_step": ckpt.get("step"),
        "val_em": ckpt.get("val_em"), "val_loss": ckpt.get("val_loss"),
    }
    run_metrics = Path(ckpt_path).parent / "metrics.json"
    if run_metrics.exists():
        with open(run_metrics, encoding="utf-8") as f:
            rm = json.load(f)
        info["run"] = {k: rm[k] for k in RUN_METRIC_KEYS if k in rm}

    if run_eval:
        # 내보낸 파일을 다시 불러 평가한다: 배포 폴더만으로 같은 결과가 나오는지 확인하는 셈이다
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_dtype = torch.bfloat16 if device.type == "cuda" else None
        model, _ = load_model(out / "model.pt", device)
        released_tok = bpe.Tokenizer.load(out / "tokenizer.json")
        info["params"] = model.num_params()
        rules = load_rules(stage)
        info["eval"] = {}
        for name in list_sets(eval_dir):
            r = evaluate_set(model, name, released_tok, device, amp_dtype, eval_dir=eval_dir, db_path=db_path, rules=rules)
            info["eval"][name] = {"overall": r["overall"], "by_tier": r["by_tier"], "error_parts": r["error_parts"]}

    with open(out / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 체크포인트 → 배포용 모델 폴더")
    ap.add_argument("--ckpt", required=True, help="train.py 가 저장한 best.pt")
    ap.add_argument("--tokenizer", default=None, help="토크나이저 (기본: 체크포인트 폴더의 tokenizer.json, 없으면 체크포인트에 적힌 경로)")
    ap.add_argument("--db", default=str(config.DB_PATH), help=f"평가의 실행 정확도용 DB (기본 {config.DB_PATH})")
    ap.add_argument("--out", default=str(config.MODEL_DIR), help=f"배포 폴더 (기본 {config.MODEL_DIR})")
    ap.add_argument("--eval-dir", default=str(config.EVAL_DIR), help=f"평가셋 상위 폴더 (기본 {config.EVAL_DIR})")
    ap.add_argument("--no-eval", action="store_true", help="내보낸 모델 평가를 생략")
    args = ap.parse_args()

    m = release(args.ckpt, args.out, run_eval=not args.no_eval, eval_dir=args.eval_dir, db_path=args.db,
                tokenizer_path=args.tokenizer)
    print(f"{args.out}/ ← {m['source_checkpoint']} (stage {m['stage']}, epoch {m['best_epoch']}, val EM {m['val_em']:.4f})")
    print(f"  model.pt, tokenizer.json (vocab {m['tokenizer']['vocab_size']}), model_info.json")
    for name, e in m.get("eval", {}).items():
        o = e["overall"]
        print(f"  {name:<14} n={o['n']:>5}  EM {o['em']:6.1%}  multi {o['em_multi']:6.1%}  exec {o['exec_acc']:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
