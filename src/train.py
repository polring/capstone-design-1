"""
train.py — 1단계 Text-to-SQL 모델 학습 루프 (계획서 3-3 1단계 6번)

저장소 루트에서 실행한다 (경로는 src/config.py, cwd 기준 상대 경로).

    python -m src.train                          # 본 학습 (config.py 설정, 이름 교체 40% 를 epoch 마다 적용)
    python -m src.train --name-swap-ratio 0      # 이름 교체 없이 원본 데이터로만 학습 (기준 실행)
    python -m src.train --overfit 100            # 과적합 테스트: 예시 100개로 EM 100% 도달 확인
    python -m src.train --ffn-dim 704 --run-name ffn704   # config 를 건드리지 않고 FFN 크기만 바꿔 비교
    python -m src.train --train-files data_raw/stage1/name_swap_30/train_pairs.json --run-name name_swap_30
    python -m src.train --name-swap-ratio 0.3    # 교체 비율 바꾸기
    python -m src.train --train-files a.json b.json --tokenizer t.json --eval-dir <폴더> --db <shop.db>   # 입력 바꾸기
    python -m src.train --eval-sets indist holdout --out <폴더>   # 최종 평가할 평가셋 고르기, 실행 폴더 직접 지정

- 입력 파일은 모두 인자로 바꿀 수 있고, 생략하면 src/config.py 의 기본값을 쓴다.

- 검증셋: 학습 쌍(--train-files, 기본 config.TRAIN_DIR/*.json)에서 SQL 단위로 val_ratio 만큼 떼어 낸다. 같은 SQL 의 질문 5개가
  train/val 에 나뉘어 들어가면 val 점수가 암기 점수가 되기 때문이다. eval_indist 는 조기 종료·
  체크포인트 선택에 쓰지 않고, 학습이 끝난 뒤 최종 측정에만 한 번 쓴다 (미등장 값 평가셋도 함께).
- 최종 측정은 src/evaluation/scoring.py 의 evaluate_set 이다 (run.py --evaluation 과 같은 함수). 저장된 체크포인트만 다시 평가할 때는 그쪽을 쓴다.
- 주 지표는 greedy 생성 결과의 Exact Match (계획서 3-4). teacher-forced 토큰 정확도는 보조 지표.
- 산출물: runs/stage<N>/<run-name>/ 에 best.pt, metrics.json, eval_<set>.json, TensorBoard 로그.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.data.dataset import PAD_ID, TextToSQLDataset, collate_fn, load_train_pairs, tokenize_pairs
from src.data.stage1.name_swap import NameSwapper, load_english_words
from src.evaluation.eval_sets import list_sets
from src.evaluation.scoring import evaluate_set, greedy_exact_match, print_report
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

DEFAULT_NAME_SWAP_RATIO = 0.4  # 진행 보고서 2026-10-02: seed 3개로 재현 확인


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def split_by_sql(pairs: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """SQL 단위로 train/val 을 나눈다 (같은 SQL 의 질문들은 한쪽에만 들어간다).

    이름 교체 데이터(name_swap.py)는 바꾸기 전 SQL(orig_sql)을 기준으로 나누고, 검증에는 원래 쌍을 쓴다.
    그래서 어떤 학습 파일을 쓰든 검증셋은 원본 데이터 기준 실행과 같다.
    """
    key = lambda p: p.get("orig_sql", p["sql"])
    sqls = sorted({key(p) for p in pairs})
    random.Random(seed).shuffle(sqls)
    val_sqls = set(sqls[: int(len(sqls) * val_ratio)])
    train = [p for p in pairs if key(p) not in val_sqls]
    val = [{"question": p.get("orig_question", p["question"]), "sql": key(p)}
           for p in pairs if key(p) in val_sqls]
    return train, val


# ---------------------------------------------------------------------------
# 학습 보조
# ---------------------------------------------------------------------------

def masked_loss(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """SQL 위치(loss_mask=True)만으로 평균 cross entropy 를 낸다. (loss, 맞힌 토큰 수) 반환."""
    loss_tok = F.cross_entropy(logits.float().flatten(0, 1), target_ids.flatten(), reduction="none")
    mask = loss_mask.flatten().float()
    loss = (loss_tok * mask).sum() / mask.sum()
    correct = ((logits.argmax(-1) == target_ids) & loss_mask).sum()
    return loss, correct


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """2D 이상 파라미터(행렬·임베딩)에만 weight decay 를 준다. RMSNorm weight 는 제외."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available(),
    )


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """선형 warmup 후 cosine 으로 max_lr * min_ratio 까지 감소."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_teacher_forced(model, loader, device, amp_dtype) -> tuple[float, float]:
    """teacher forcing 상태의 SQL 토큰 평균 loss 와 토큰 정확도."""
    model.eval()
    total_loss, total_tok, total_correct = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(batch["input_ids"])
        loss, correct = masked_loss(logits, batch["target_ids"], batch["loss_mask"])
        n = batch["loss_mask"].sum().item()
        total_loss += loss.item() * n
        total_tok += n
        total_correct += correct.item()
    model.train()
    return total_loss / total_tok, total_correct / total_tok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="1단계 Text-to-SQL 학습")
    ap.add_argument("--train-files", nargs="+", default=None,
                    help=f"학습 쌍 JSON 파일들 (기본: {config.TRAIN_DIR}/*.json 전체). 예: name_swap.py 출력")
    ap.add_argument("--tokenizer", default=str(config.TOKENIZER_PATH), help=f"토크나이저 (기본 {config.TOKENIZER_PATH})")
    ap.add_argument("--eval-dir", default=str(config.EVAL_DIR), help=f"학습 후 최종 평가할 평가셋 폴더 (기본 {config.EVAL_DIR})")
    ap.add_argument("--eval-sets", nargs="+", default=None, help="학습 후 최종 평가할 평가셋 이름들 (기본: --eval-dir 의 전부)")
    ap.add_argument("--out", default=None, help=f"실행 폴더 (기본: {config.RUNS_DIR}/<run-name>/)")
    ap.add_argument("--db", default=str(config.DB_PATH), help=f"최종 평가의 실행 정확도용 DB (기본 {config.DB_PATH})")
    ap.add_argument("--name-swap-ratio", type=float, default=None,
                    help="epoch 마다 이름 조건 쌍의 이 비율을 새 가짜 이름으로 바꿔 학습 (name_swap.py). 0 이면 끔. "
                         f"기본 {DEFAULT_NAME_SWAP_RATIO} (이미 이름을 바꾼 파일이거나 --overfit 이면 0)")
    ap.add_argument("--wordlist", default=str(config.WORDLIST_PATH), help="--name-swap-ratio 용 영단어 목록")
    ap.add_argument("--run-name", default=None, help=f"{config.RUNS_DIR}/<run-name>/ 에 저장 (기본: 설정에서 자동 생성)")
    ap.add_argument("--ffn-dim", type=int, default=config.FFN_DIM)
    ap.add_argument("--epochs", type=int, default=25, help="상한 (조기 종료 가능, 계획서 3-2)")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--val-em-samples", type=int, default=0, help="epoch 마다 EM 을 볼 val 예시 수 (0 = 전부)")
    ap.add_argument("--patience", type=int, default=10, help="val EM 개선 없는 epoch 수 한도")
    ap.add_argument("--overfit", type=int, default=0, help="N>0 이면 학습 예시 N개로 과적합 테스트")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기")
    args = ap.parse_args()
    if args.eval_sets:  # 10분 학습 뒤에 실패하지 않도록 평가셋 이름을 먼저 확인한다
        unknown = sorted(set(args.eval_sets) - set(list_sets(args.eval_dir)))
        if unknown:
            ap.error(f"{args.eval_dir} 에 없는 평가셋: {unknown} (있는 것: {list_sets(args.eval_dir)})")
    all_train = load_train_pairs(args.train_files)
    # name_swap.py 로 이미 이름을 바꾼 파일(orig_sql 이 있음)은 다시 바꾸지 않는다
    pre_swapped = any("orig_sql" in p for p in all_train)
    if args.name_swap_ratio is None:
        args.name_swap_ratio = 0.0 if (pre_swapped or args.overfit) else DEFAULT_NAME_SWAP_RATIO
    elif args.name_swap_ratio and (pre_swapped or args.overfit):
        ap.error("--name-swap-ratio 는 이미 이름을 바꾼 학습 파일 / --overfit 과 함께 쓰지 않는다")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None

    # --- 데이터 ---
    tok = bpe.Tokenizer.load(args.tokenizer)

    if args.overfit:
        # 과적합 테스트: 같은 N개로 학습·평가. 일반화가 아니라 파이프라인 버그 여부를 본다.
        pairs = random.Random(args.seed).sample(all_train, args.overfit)
        train_ex = val_ex = tokenize_pairs(pairs, tok)
        args.weight_decay = 0.0
        args.patience = 10**9
        eval_every = max(1, 400 // math.ceil(len(train_ex) / args.batch_size))  # 약 400 step 마다
        args.epochs = args.epochs if args.epochs != 25 else 2000
    else:
        train_pairs, val_pairs = split_by_sql(all_train, args.val_ratio, args.seed)
        train_ex = tokenize_pairs(train_pairs, tok)
        val_ex = tokenize_pairs(val_pairs, tok)
        eval_every = 1

    def make_loader(examples):
        return DataLoader(TextToSQLDataset(examples), batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, drop_last=not args.overfit, pin_memory=True)

    # epoch 마다 이름 교체: 검증셋은 위에서 원본 기준으로 이미 떼었고, 학습 쌍만 매 epoch 새로 바꾼다.
    # 쌍 수가 그대로라 epoch 당 step 수(= lr 스케줄)는 바뀌지 않는다.
    swapper = None
    if args.name_swap_ratio:
        swapper = NameSwapper(all_train, load_english_words(args.wordlist))
        swap_rng = random.Random(args.seed)
    loader = make_loader(train_ex)
    val_loader = DataLoader(TextToSQLDataset(val_ex), batch_size=256, shuffle=False,
                            collate_fn=collate_fn)
    em_examples = val_ex
    if args.val_em_samples and args.val_em_samples < len(val_ex):
        em_examples = random.Random(args.seed).sample(val_ex, args.val_em_samples)

    # --- 모델 ---
    model = TextToSQLModel(ffn_dim=args.ffn_dim, padding_idx=PAD_ID).to(device)
    n_params = model.num_params()
    opt = make_optimizer(model, args.lr, args.weight_decay)
    total_steps = args.epochs * len(loader)

    swap_tag = f"_swap{round(args.name_swap_ratio * 100)}" if args.name_swap_ratio else ""
    run_name = args.run_name or (f"overfit{args.overfit}_ffn{args.ffn_dim}" if args.overfit
                                 else f"ffn{args.ffn_dim}{swap_tag}")
    run_dir = Path(args.out) if args.out else config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # 학습에 쓴 토크나이저 사본: config.TOKENIZER_PATH 가 다시 만들어져도 이 체크포인트를 평가할 수 있게 한다
    tokenizer_copy = run_dir / "tokenizer.json"
    tok.save(tokenizer_copy)
    writer = SummaryWriter(str(run_dir / "tb"))
    model_cfg = dict(vocab_size=config.VOCAB_SIZE, dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                     num_heads=config.NUM_HEADS, ffn_dim=args.ffn_dim, max_seq_len=config.SEQ_LEN,
                     padding_idx=PAD_ID)

    print(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) "
          f"amp={amp_dtype}")
    print(f"params={n_params:,}  ffn_dim={args.ffn_dim}  train={len(train_ex):,}  val={len(val_ex):,}  "
          f"steps/epoch={len(loader)}  max_steps={total_steps:,}")

    # --- 학습 ---
    step, best_em, best_val_loss, bad_epochs = 0, -1.0, float("inf"), 0
    history: list[dict] = []
    tokens_seen, train_time = 0, 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        epoch_tokens = 0
        if swapper:
            swapped, swap_stats = swapper.swap(train_pairs, args.name_swap_ratio, swap_rng)
            # 바뀐 쌍만 다시 토큰화한다 (순수 Python BPE 라 전체를 다시 하면 epoch 당 1분 넘게 걸린다)
            loader = make_loader([tokenize_pairs([p], tok)[0] if p["swapped"] else ex
                                  for p, ex in zip(swapped, train_ex)])
            if epoch == 1:
                print(f"name swap (epoch 마다): {swap_stats}")
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            lr = lr_at(step, args.lr, args.warmup, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(batch["input_ids"])
            loss, _ = masked_loss(logits, batch["target_ids"], batch["loss_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_tokens += batch["input_ids"].numel()  # 패딩 포함: 실제 연산한 토큰 수
            if step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_time += dt
        tokens_seen += epoch_tokens

        if epoch % eval_every and epoch != args.epochs:
            continue

        val_loss, val_tok_acc = evaluate_teacher_forced(model, val_loader, device, amp_dtype)
        val_em, _ = greedy_exact_match(model, em_examples, tok, device, amp_dtype)
        tok_per_s = epoch_tokens / dt
        flops_per_s = 6 * n_params * tok_per_s
        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("val/token_acc", val_tok_acc, step)
        writer.add_scalar("val/em", val_em, step)
        writer.add_scalar("perf/tokens_per_s", tok_per_s, step)
        history.append(dict(epoch=epoch, step=step, train_loss=loss.item(), val_loss=val_loss,
                            val_token_acc=val_tok_acc, val_em=val_em, tokens_per_s=tok_per_s))
        print(f"epoch {epoch:4d} step {step:6d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
              f"tok_acc {val_tok_acc:.4f} EM {val_em:.4f} | {tok_per_s / 1e3:.0f}k tok/s "
              f"({flops_per_s:.2e} FLOPs/s) {dt:.1f}s/epoch")

        improved = val_em > best_em or (val_em == best_em and val_loss < best_val_loss)
        if improved:
            best_em, best_val_loss, bad_epochs = val_em, val_loss, 0
            torch.save({"model": model.state_dict(), "model_cfg": model_cfg, "stage": config.STAGE,
                        "tokenizer_path": str(tokenizer_copy), "epoch": epoch, "step": step,
                        "val_em": val_em, "val_loss": val_loss, "args": vars(args)},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
        if args.overfit and val_em == 1.0:
            print(f"과적합 테스트 통과: {epoch} epoch / {step} step 에서 EM 100%")
            break
        if bad_epochs >= args.patience:
            print(f"조기 종료: {args.patience} epoch 동안 val EM 개선 없음")
            break

    total_time = time.time() - t_start
    result = dict(run_name=run_name, params=n_params, ffn_dim=args.ffn_dim, epochs_run=epoch, steps=step,
                  best_val_em=best_em, best_val_loss=best_val_loss, train_time_s=train_time,
                  total_time_s=total_time, avg_tokens_per_s=tokens_seen / train_time,
                  avg_flops_per_s=6 * n_params * tokens_seen / train_time, history=history)

    # --- 최종 측정: best 체크포인트로 평가셋 전부 (과적합 테스트에서는 생략) ---
    if not args.overfit:
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        result["best_epoch"] = ckpt["epoch"]
        print(f"best epoch {ckpt['epoch']} (val EM {ckpt['val_em']:.4f})")
        for name in args.eval_sets or list_sets(args.eval_dir):
            ev = evaluate_set(model, name, tok, device, amp_dtype, eval_dir=args.eval_dir, db_path=args.db)
            print_report(ev)
            result[f"eval_{name}"] = {k: v for k, v in ev.items() if k != "failures"}
            with open(run_dir / f"eval_{name}.json", "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False, indent=2)

    print(f"학습 시간 {train_time:.0f}s (전체 {total_time:.0f}s), "
          f"평균 {result['avg_tokens_per_s'] / 1e3:.0f}k tok/s, {result['avg_flops_per_s']:.2e} FLOPs/s")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
