# 역할: 질문에서 SQL을 생성하는 모델의 학습·검증·체크포인트 저장을 실행한다.
# 사용처: python -m src.train으로 단계별 모델을 처음부터 학습하거나 학습을 이어갈 때 사용한다.

"""
train.py — 1·2·3단계 공통 Text-to-SQL 모델 학습 루프

저장소 루트에서 실행한다 (경로는 src/config.py, cwd 기준 상대 경로).

    python -m src.train                          # 본 학습 (config.py 설정, 이름 교체 40% 를 epoch 마다 적용)
    python -m src.train --name-swap-ratio 0      # 이름 교체 없이 원본 데이터로만 학습 (기준 실행)
    python -m src.train --overfit 100            # 과적합 테스트: 예시 100개로 EM 100% 도달 확인
    python -m src.train --ffn-dim 704 --run-name ffn704   # config 를 건드리지 않고 FFN 크기만 바꿔 비교
    python -m src.train --train-files data_raw/stage1/name_swap_30/train_pairs.json --run-name name_swap_30
    python -m src.train --name-swap-ratio 0.3    # 교체 비율 바꾸기
    python -m src.train --stage 3               # 3단계 경로·모델 설정으로 새 모델 학습
    python -m src.train --llm-repeat 3           # 2단계: LLM 학습 질문을 3번씩 넣기 (유의어 교체가 epoch 마다 사본을 다르게 바꾼다)
    python -m src.train --train-files a.json b.json --tokenizer t.json --eval-dir <폴더> --db <shop.db>   # 입력 바꾸기
    python -m src.train --eval-sets indist holdout --out <폴더>   # 최종 평가할 평가셋 고르기, 실행 폴더 직접 지정

- 입력 파일은 모두 인자로 바꿀 수 있고, 생략하면 src/config.py 의 기본값을 쓴다.

- 검증셋: 학습 쌍(--train-files, 기본 config.TRAIN_DIR/*.json)에서 SQL·대비 묶음 단위로 val_ratio 만큼 떼어 낸다. 같은 SQL 의 질문 5개가
  train/val 에 나뉘어 들어가면 val 점수가 암기 점수가 되기 때문이다. eval_indist 는 조기 종료·
  체크포인트 선택에 쓰지 않고, 학습이 끝난 뒤 최종 측정에만 한 번 쓴다 (미등장 값 평가셋도 함께).
- 최종 측정은 src/evaluation/scoring.py 의 evaluate_set 이다 (run.py --evaluation 과 같은 함수). 저장된 체크포인트만 다시 평가할 때는 그쪽을 쓴다.
- 주 지표는 greedy 생성 결과의 Exact Match (계획서 3-4). teacher-forced 토큰 정확도는 보조 지표.
- 산출물: runs/stage<N>/<run-name>/ 에 best.pt, metrics.json, eval_<set>.json, TensorBoard 로그.
"""

from __future__ import annotations

import argparse
import importlib
from itertools import islice
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
from src.data.stage1.name_swap import load_english_words
from src.data.splitting import split_by_sql, validate_split
from src.evaluation.eval_sets import list_sets
from src.evaluation.scoring import evaluate_set, greedy_exact_match, print_report, load_rules
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

DEFAULT_NAME_SWAP_RATIO = 0.4  # 진행 보고서 2026-10-02: seed 3개로 재현 확인


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def parse_round_repeat(value: str) -> tuple[str, int]:
    """Parse an explicit LLM round repetition override, e.g. round_name=1."""
    name, sep, count = value.rpartition("=")
    try:
        repeat = int(count)
    except ValueError:
        repeat = 0
    if not sep or not name.strip() or repeat < 1:
        raise argparse.ArgumentTypeError("expected ROUND=N with a nonempty round and positive integer N")
    return name, repeat


def repeat_llm_pairs(pairs: list[dict], repeat: int, round_repeats: dict[str, int] | None = None) -> list[dict]:
    """Apply repetition after splitting; retain old ordering when no override is set."""
    def is_llm(pair):
        return str(pair.get("source", "")).startswith("llm:")

    overrides = round_repeats or {}
    if repeat < 1 or any(n < 1 for n in overrides.values()):
        raise ValueError("LLM repetition counts must be positive")
    rounds = {p.get("round") for p in pairs if is_llm(p)}
    unknown = sorted(set(overrides) - rounds)
    if unknown:
        raise ValueError(f"LLM rounds absent from training: {unknown}")
    weights = [overrides.get(p.get("round"), repeat) if is_llm(p) else 1 for p in pairs]
    result = list(pairs)
    for copy in range(1, max(weights, default=1)):
        result.extend(p for p, n in zip(pairs, weights) if n > copy)
    return result


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

def main(argv: list[str] | None = None) -> int:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--stage", type=int, choices=[1, 2, 3], default=config.STAGE)
    selected, _ = selector.parse_known_args(argv)
    stage = selected.stage
    paths = config.stage_paths(stage)
    preset = config.STAGE_PRESETS[stage]
    rules = load_rules(stage)
    ap = argparse.ArgumentParser(description="단계 공통 Text-to-SQL 학습", parents=[selector])
    ap.add_argument("--train-files", nargs="+", default=None,
                    help=f"학습 쌍 JSON·gzip JSON (기본: {paths.train_dir}의 *.json·*.json.gz). 예: name_swap.py 출력")
    ap.add_argument("--val-files", nargs="+", default=None,
                    help="미리 분리한 검증 쌍 JSON. 지정하면 --train-files 도 필요하며 --val-ratio 로 다시 나누지 않는다")
    ap.add_argument("--tokenizer", default=str(paths.tokenizer_path), help="학습에 사용할 토크나이저")
    ap.add_argument("--eval-dir", default=str(paths.eval_dir), help="학습 후 최종 평가할 평가셋 폴더")
    ap.add_argument("--eval-sets", nargs="+", default=None, help="학습 후 최종 평가할 평가셋 이름들 (기본: --eval-dir 의 전부)")
    ap.add_argument("--out", default=None, help=f"실행 폴더 (기본: {paths.runs_dir}/<run-name>/)")
    ap.add_argument("--db", default=str(paths.db_path), help="실행 정확도용 DB")
    ap.add_argument("--name-pools", default=str(paths.data_dir / "name_pools.json"), help="3단계 이름·상품 교체 풀")
    ap.add_argument("--name-swap-ratio", type=float, default=None,
                    help="epoch 마다 이름 조건 쌍의 이 비율을 새 가짜 이름으로 바꿔 학습 (name_swap.py). 0 이면 끔. "
                         f"기본 {DEFAULT_NAME_SWAP_RATIO} (이미 이름을 바꾼 파일이거나 --overfit 이면 0)")
    ap.add_argument("--like-sampling", choices=["markov", "balanced"], default="markov",
                    help="3단계 LIKE 교체 조각 선택: 기존 markov 또는 학습 SQL 토큰 빈도별 균형 balanced. "
                         "교체 확률은 --name-swap-ratio 를 그대로 사용")
    ap.add_argument("--wordlist", default=str(config.WORDLIST_PATH), help="--name-swap-ratio 용 영단어 목록")
    ap.add_argument("--llm-synonym-prob", type=float, default=None,
                    help="2·3단계: epoch 마다 LLM 학습 질문의 일상 단어를 조합기 유사어로 바꾸는 확률. 0 이면 끔. "
                         "기본: 그 단계 synonyms.json 의 prob (--overfit 이면 비활성화)")
    ap.add_argument("--composer-synonym-prob", type=float, default=0.0,
                    help="2·3단계: epoch 마다 조합기 학습 문장(source composer*)의 일상 단어를 유사어로 바꾸는 확률. "
                         "기본 0 (끔). 조합기 문장은 만들 때 이미 한 번 바꿨으므로 epoch 마다 다른 대체어를 보여 준다")
    # 3단계는 2단계 최종 설정(stage2_llm3: LLM 쌍 3번 + epoch 마다 유사어 교체)을 기본으로 쓴다
    ap.add_argument("--llm-repeat", type=int, default=3 if stage >= 3 else 1,
                    help="2·3단계: LLM 학습 질문(source llm:*)을 학습 쌍에 넣는 횟수 (기본 2단계 1, 3단계 3)")
    ap.add_argument("--llm-round-repeat", action="append", type=parse_round_repeat, default=[], metavar="ROUND=N",
                    help="특정 LLM 라운드의 반복 횟수만 지정. 예: --llm-round-repeat llm_numbers_r6_600=1 (여러 번 지정 가능)")
    ap.add_argument("--run-name", default=None, help=f"{paths.runs_dir}/<run-name>/ 에 저장 (기본: 설정에서 자동 생성)")
    ap.add_argument("--ffn-dim", type=int, default=None, help="기본: round(8 * dim / 3) (config.FFN_DIM)")
    # 모델 크기 실험용 (기본값은 config 의 단계 프리셋)
    ap.add_argument("--layers", type=int, default=preset["NUM_LAYERS"])
    ap.add_argument("--dim", type=int, default=preset["HIDDEN_DIM"])
    ap.add_argument("--heads", type=int, default=preset["NUM_HEADS"])
    ap.add_argument("--seq-len", type=int, default=preset["SEQ_LEN"])
    ap.add_argument("--model-vocab-size", type=int, default=preset["VOCAB_SIZE"], help="임베딩 용량. 출력은 실제 토크나이저 어휘로 제한")
    ap.add_argument("--epochs", type=int, default=None, help="상한 (기본 1·2단계 25, 3단계 15, overfit 2000)")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--steps-per-epoch", type=int, default=None,
                    help="epoch마다 사용할 학습 배치 수. 데이터량이 다른 실험의 총 step과 lr 스케줄을 맞출 때 사용")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--val-em-samples", type=int, default=0, help="epoch 마다 EM 을 볼 val 예시 수 (0 = 전부)")
    ap.add_argument("--patience", type=int, default=10, help="val EM 개선 없는 epoch 수 한도")
    ap.add_argument("--save-every", type=int, default=0,
                    help="N>0 이면 N epoch 마다 epoch<N>.pt 도 저장 (학습량에 따른 평가 추이 확인용)")
    ap.add_argument("--init", default=None,
                    help="이어 학습: 이 체크포인트(best.pt / epoch<N>.pt)의 가중치로 시작. 옵티마이저·lr 스케줄은 새로 시작하므로 "
                         "--lr 를 낮춰 쓴다. 모델 설정과 토크나이저가 같아야 한다")
    ap.add_argument("--resume", default=None,
                    help="중단된 학습 이어 하기: 이 체크포인트 다음 epoch 부터 같은 lr 스케줄로 계속한다. 처음과 같은 인자로 실행한다. "
                         "last.pt 는 옵티마이저·난수 상태까지 복원하고, best.pt / epoch<N>.pt 는 가중치만 있어 AdamW 상태를 새로 시작한다")
    ap.add_argument("--overfit", type=int, default=0, help="N>0 이면 학습 예시 N개로 과적합 테스트")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true", help="bf16 autocast 끄기")
    ap.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU 사용")
    args = ap.parse_args(argv)
    default_round_repeats = {}
    # A consolidated stage owns its frozen split; do not silently split it again.
    if not args.train_files and not args.overfit and paths.validation_path.is_file():
        args.train_files = [str(paths.train_pairs_path)]
        if args.val_files is None:
            args.val_files = [str(paths.validation_path)]
        recipe_path = paths.data_dir / "dataset_info.json"
        if recipe_path.is_file():
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            default_round_repeats = recipe.get("training_defaults", {}).get("round_repeats", {})
    if len(dict(args.llm_round_repeat)) != len(args.llm_round_repeat):
        ap.error("--llm-round-repeat must not repeat the same round")
    args.llm_round_repeat = {**default_round_repeats, **dict(args.llm_round_repeat)}
    if args.like_sampling == "balanced" and (stage != 3 or args.overfit):
        ap.error("--like-sampling balanced 는 3단계 본 학습에서만 사용한다")
    if args.resume and (args.init or args.overfit):
        ap.error("--resume 은 --init / --overfit 과 함께 쓰지 않는다")
    if args.val_files and (not args.train_files or args.overfit):
        ap.error("--val-files requires --train-files and cannot be used with --overfit")
    if args.epochs is None:
        args.epochs = 2000 if args.overfit else 15 if stage >= 3 else 25
    if args.llm_synonym_prob is not None and not 0 <= args.llm_synonym_prob <= 1:
        ap.error("--llm-synonym-prob must be within 0..1")
    if not 0 <= args.composer_synonym_prob <= 1:
        ap.error("--composer-synonym-prob must be within 0..1")
    if args.llm_repeat < 1 or args.epochs < 1:
        ap.error("--llm-repeat and --epochs must be positive")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        ap.error("--steps-per-epoch must be positive")
    if args.overfit and args.steps_per_epoch is not None:
        ap.error("--steps-per-epoch 는 --overfit 과 함께 쓰지 않는다")
    if args.ffn_dim is None:
        args.ffn_dim = round(args.dim * 8 / 3)
    if args.eval_sets:  # 10분 학습 뒤에 실패하지 않도록 평가셋 이름을 먼저 확인한다
        unknown = sorted(set(args.eval_sets) - set(list_sets(args.eval_dir)))
        if unknown:
            ap.error(f"{args.eval_dir} 에 없는 평가셋: {unknown} (있는 것: {list_sets(args.eval_dir)})")
    all_train = load_train_pairs(args.train_files, train_dir=paths.train_dir)
    # name_swap.py 로 이미 이름을 바꾼 파일(orig_sql 이 있음)은 다시 바꾸지 않는다
    pre_swapped = any("orig_sql" in p for p in all_train)
    if args.name_swap_ratio is None:
        args.name_swap_ratio = 0.0 if (pre_swapped or args.overfit) else DEFAULT_NAME_SWAP_RATIO
    elif args.name_swap_ratio and (pre_swapped or args.overfit):
        ap.error("--name-swap-ratio 는 이미 이름을 바꾼 학습 파일 / --overfit 과 함께 쓰지 않는다")
    if args.like_sampling == "balanced" and not 0 < args.name_swap_ratio <= 1:
        ap.error("--like-sampling balanced 는 0 < --name-swap-ratio <= 1 이 필요하다")
    if stage < 2 and (args.llm_synonym_prob or args.composer_synonym_prob or args.llm_repeat != 1 or args.llm_round_repeat):
        ap.error("--llm-synonym-prob / --composer-synonym-prob / --llm-repeat / --llm-round-repeat 는 2단계부터 쓴다")
    if args.overfit and args.llm_round_repeat:
        ap.error("--llm-round-repeat 는 --overfit 과 함께 쓰지 않는다")
    if args.overfit and args.composer_synonym_prob:
        ap.error("--composer-synonym-prob 는 --overfit 과 함께 쓰지 않는다")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None

    # --- 데이터 ---
    tok = bpe.Tokenizer.load(args.tokenizer)
    if args.model_vocab_size < tok.vocab_size:
        ap.error("--model-vocab-size must cover the tokenizer vocabulary")

    if args.overfit:
        # 과적합 테스트: 같은 N개로 학습·평가. 일반화가 아니라 파이프라인 버그 여부를 본다.
        pairs = random.Random(args.seed).sample(all_train, args.overfit)
        train_ex = val_ex = tokenize_pairs(pairs, tok)
        args.weight_decay = 0.0
        args.patience = 10**9
        eval_every = max(1, 400 // math.ceil(len(train_ex) / args.batch_size))  # 약 400 step 마다
    else:
        if args.val_files:
            train_pairs, val_pairs = all_train, load_train_pairs(args.val_files)
            try:
                validate_split(train_pairs, val_pairs)
            except ValueError as exc:
                ap.error(str(exc))
            if not train_pairs or not val_pairs:
                ap.error("explicit train/validation files must both contain pairs")
        else:
            train_pairs, val_pairs = split_by_sql(all_train, args.val_ratio, args.seed)
        # 검증셋을 뗀 뒤 늘린다. 특정 라운드만 낮춰도 다른 LLM 데이터의 반복은 유지한다.
        try:
            train_pairs = repeat_llm_pairs(train_pairs, args.llm_repeat, args.llm_round_repeat)
        except ValueError as exc:
            ap.error(str(exc))
        if args.llm_round_repeat:
            print(f"LLM round repetition overrides: {args.llm_round_repeat}")
        train_ex = tokenize_pairs(train_pairs, tok)
        val_ex = tokenize_pairs(val_pairs, tok)
        # 모델 최대 길이를 넘는 쌍은 위치 임베딩이 없어 학습할 수 없다 (2단계 조합기 문장 중 극소수)
        too_long = lambda ex: len(ex["ids"]) - 1 > args.seq_len
        n_long = sum(map(too_long, train_ex)) + sum(map(too_long, val_ex))
        if n_long:
            print(f"최대 길이 {args.seq_len} 을 넘는 쌍 {n_long}개 제외")
            keep = [not too_long(ex) for ex in train_ex]
            train_pairs = [p for p, k in zip(train_pairs, keep) if k]
            train_ex = [ex for ex, k in zip(train_ex, keep) if k]
            val_ex = [ex for ex in val_ex if not too_long(ex)]
        eval_every = 1

    def make_loader(examples):
        return DataLoader(TextToSQLDataset(examples), batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, drop_last=not args.overfit, pin_memory=True)

    # epoch 마다 이름 교체: 검증셋은 위에서 원본 기준으로 이미 떼었고, 학습 쌍만 매 epoch 새로 바꾼다.
    # 쌍 수가 그대로라 epoch 당 step 수(= lr 스케줄)는 바뀌지 않는다.
    syn_swapper = None
    if stage >= 2 and not args.overfit and (args.llm_synonym_prob != 0 or args.composer_synonym_prob):
        SynonymSwapper = importlib.import_module(f"src.data.stage{stage}.compose").SynonymSwapper
        syn_swapper = SynonymSwapper()
        if args.llm_synonym_prob is None:
            args.llm_synonym_prob = syn_swapper.prob
        syn_rng = random.Random(args.seed + 1)
    swapper = None
    like_sampler = None
    if args.name_swap_ratio:
        # 2단계부터는 OR 그룹의 모든 이름과 LIKE 조각까지 바꾼다 (src/data/stage2/name_swap.py)
        NameSwapper = importlib.import_module(f"src.data.stage{stage}.name_swap").NameSwapper
        if args.like_sampling == "balanced":
            from src.data.db_gen import load_bans
            from src.data.stage3.like_sampling import LikeFragmentSampler
            bans = load_bans(json.loads(paths.holdout_path.read_text(encoding="utf-8")))
            banned = {value.strip("%") for table, col in (("customers", "name"), ("items", "item_name"))
                      for value in bans.get(table, {}).get(col, set()) if isinstance(value, str)}
            # After validation splitting, LLM repetition and the initial length filter.
            like_sampler = LikeFragmentSampler(train_pairs, tok, banned_fragments=banned)
            print(f"LIKE sampling: {like_sampler.describe()}")
        swapper = (NameSwapper(all_train, db=args.db, pools_path=args.name_pools, like_sampler=like_sampler) if stage >= 3
                   else NameSwapper(all_train, load_english_words(args.wordlist)))
        swap_rng = random.Random(args.seed)
    loader = make_loader(train_ex)
    val_loader = DataLoader(TextToSQLDataset(val_ex), batch_size=256, shuffle=False,
                            collate_fn=collate_fn)
    em_examples = val_ex
    if args.val_em_samples and args.val_em_samples < len(val_ex):
        em_examples = random.Random(args.seed).sample(val_ex, args.val_em_samples)

    # --- 모델 ---
    model = TextToSQLModel(dim=args.dim, num_layers=args.layers, num_heads=args.heads, ffn_dim=args.ffn_dim,
                           vocab_size=args.model_vocab_size, valid_vocab_size=tok.vocab_size,
                           max_seq_len=args.seq_len, padding_idx=PAD_ID).to(device)
    n_params = model.num_params()
    opt = make_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = args.steps_per_epoch or len(loader)
    if steps_per_epoch > len(loader):
        ap.error(f"--steps-per-epoch {steps_per_epoch} exceeds available batches {len(loader)}")
    total_steps = args.epochs * steps_per_epoch

    swap_tag = f"_swap{round(args.name_swap_ratio * 100)}" if args.name_swap_ratio else ""
    like_tag = "_like_balanced" if args.like_sampling == "balanced" else ""
    run_name = args.run_name or (f"overfit{args.overfit}_ffn{args.ffn_dim}" if args.overfit
                                 else f"ffn{args.ffn_dim}{swap_tag}{like_tag}")
    run_dir = Path(args.out) if args.out else paths.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # 학습에 쓴 토크나이저 사본: config.TOKENIZER_PATH 가 다시 만들어져도 이 체크포인트를 평가할 수 있게 한다
    tokenizer_copy = run_dir / "tokenizer.json"
    tok.save(tokenizer_copy)
    writer = SummaryWriter(str(run_dir / "tb"))
    model_cfg = dict(vocab_size=args.model_vocab_size, valid_vocab_size=tok.vocab_size, dim=args.dim, num_layers=args.layers,
                     num_heads=args.heads, ffn_dim=args.ffn_dim, max_seq_len=args.seq_len,
                     padding_idx=PAD_ID)
    def load_start_ckpt(path: str, flag: str) -> dict:
        # last.pt 에는 난수 상태(Python tuple)가 있어 weights_only 로 못 읽는다. 이 학습기가 저장한 파일만 넘긴다
        ckpt = torch.load(path, map_location=device, weights_only=flag == "--init")
        if ckpt["model_cfg"] != model_cfg:
            raise SystemExit(f"{flag} 모델 설정이 다르다: {ckpt['model_cfg']} != {model_cfg}")
        ckpt_tok = Path(path).parent / "tokenizer.json"  # 체크포인트 옆에 저장된 학습 당시 토크나이저
        if ckpt_tok.exists() and ckpt_tok.read_bytes() != Path(args.tokenizer).read_bytes():
            raise SystemExit(f"{flag} 토크나이저가 다르다: {ckpt_tok} != {args.tokenizer}")
        model.load_state_dict(ckpt["model"])
        return ckpt

    if args.init:
        init = load_start_ckpt(args.init, "--init")
        print(f"이어 학습: {args.init} (epoch {init['epoch']}, val EM {init['val_em']:.4f}) 의 가중치로 시작")

    start_epoch, step, best_em, best_val_loss, bad_epochs = 1, 0, -1.0, float("inf"), 0
    history: list[dict] = []
    tokens_seen, train_time = 0, 0.0
    if args.resume:
        res = load_start_ckpt(args.resume, "--resume")
        if (res["args"].get("llm_repeat", 1) != args.llm_repeat
                or res["args"].get("llm_round_repeat", {}) != args.llm_round_repeat):
            raise SystemExit("--resume 의 LLM 반복 설정이 다르다. 비중 비교는 새 학습으로 시작한다")
        if res["args"].get("like_sampling", "markov") != args.like_sampling:
            raise SystemExit("--resume 의 LIKE 선택 방식이 다르다. 방식 비교는 --init 으로 시작한다")
        if (res["args"]["epochs"] != args.epochs
                or res["args"].get("steps_per_epoch") != args.steps_per_epoch
                or res["step"] != res["epoch"] * steps_per_epoch):
            raise SystemExit(f"--resume 의 epoch 수·epoch 당 step 이 이번 실행과 다르다 (lr 스케줄이 달라진다): "
                             f"epochs {res['args']['epochs']} vs {args.epochs}, step {res['step']} vs "
                             f"{res['epoch']} x {steps_per_epoch}")
        start_epoch, step = res["epoch"] + 1, res["step"]
        if "optimizer" in res:  # last.pt: 중단 직전 상태 그대로
            opt.load_state_dict(res["optimizer"])
            best_em, best_val_loss, bad_epochs = res["best_em"], res["best_val_loss"], res["bad_epochs"]
            history, tokens_seen, train_time = res["history"], res["tokens_seen"], res["train_time"]
            rng = res["rng"]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"].cpu())  # map_location 때문에 GPU 로 올라온 난수 상태를 되돌린다
            if device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            if syn_swapper:
                syn_rng.setstate(rng["syn"])
            if swapper:
                swap_rng.setstate(rng["swap"])
        else:
            # 가중치만 있는 체크포인트: AdamW 모멘트는 새로 쌓인다. epoch 별 증강은 앞 epoch 교체를 다시 돌려
            # 끊기지 않은 학습과 같은 난수 흐름으로 맞춘다 (배치 섞는 순서는 다르다)
            best_path = run_dir / "best.pt"
            best = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else res
            best_em, best_val_loss = best["val_em"], best["val_loss"]
            for _ in range(1, start_epoch):
                replay = train_pairs
                if syn_swapper:
                    replay, _ = syn_swapper.swap(replay, args.llm_synonym_prob or 0.0, syn_rng,
                                                 composer_prob=args.composer_synonym_prob)
                if swapper:
                    swapper.swap(replay, args.name_swap_ratio, swap_rng)
            print("--resume: 옵티마이저 상태가 없어 AdamW 를 새로 시작한다")
        print(f"이어 학습: {args.resume} (epoch {res['epoch']}, step {step}) 다음 epoch {start_epoch} 부터, "
              f"best val EM {best_em:.4f}")

    if like_sampler is not None:
        (run_dir / "like_sampling.json").write_text(
            json.dumps(like_sampler.describe(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) "
          f"amp={amp_dtype}")
    print(f"params={n_params:,}  ffn_dim={args.ffn_dim}  train={len(train_ex):,}  val={len(val_ex):,}  "
          f"steps/epoch={steps_per_epoch}  max_steps={total_steps:,}")

    # --- 학습 ---
    t_start = time.time()

    epoch = start_epoch - 1  # 이미 끝난 학습을 --resume 하면 최종 평가만 다시 한다
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        epoch_tokens = 0
        if syn_swapper or swapper:
            epoch_pairs = train_pairs
            if syn_swapper:  # 유의어 교체 → 이름 교체 순서 (이름 교체는 질문의 이름만 바꾼다)
                epoch_pairs, syn_stats = syn_swapper.swap(epoch_pairs, args.llm_synonym_prob or 0.0, syn_rng,
                                                          composer_prob=args.composer_synonym_prob)
                if epoch == start_epoch:
                    print(f"유의어 교체 (epoch 마다, LLM 확률 {args.llm_synonym_prob}, "
                          f"조합기 확률 {args.composer_synonym_prob}): {syn_stats}")
            if swapper:
                epoch_pairs, swap_stats = swapper.swap(epoch_pairs, args.name_swap_ratio, swap_rng)
                if epoch == start_epoch:
                    print(f"name swap (epoch 마다): {swap_stats}")
            # 바뀐 쌍만 다시 토큰화한다 (순수 Python BPE 라 전체를 다시 하면 epoch 당 1분 넘게 걸린다).
            # 바꾼 문장이 최대 길이를 넘으면 원래 쌍을 쓴다
            def retok(p, ex):
                if not (p.get("swapped") or p.get("synonym_swapped")):
                    return ex
                new = tokenize_pairs([p], tok)[0]
                return ex if len(new["ids"]) - 1 > args.seq_len else new
            loader = make_loader([retok(p, ex) for p, ex in zip(epoch_pairs, train_ex)])
        for batch in islice(loader, steps_per_epoch):
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

        ckpt = {"model": model.state_dict(), "model_cfg": model_cfg, "stage": stage,
                "tokenizer_path": str(tokenizer_copy), "epoch": epoch, "step": step,
                "val_em": val_em, "val_loss": val_loss, "args": vars(args)}
        if args.save_every and epoch % args.save_every == 0:
            torch.save(ckpt, run_dir / f"epoch{epoch}.pt")
        improved = val_em > best_em or (val_em == best_em and val_loss < best_val_loss)
        if improved:
            best_em, best_val_loss, bad_epochs = val_em, val_loss, 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad_epochs += 1
        if not args.overfit:  # --resume 용: 다음 epoch 를 끊기지 않은 학습과 같은 상태로 시작할 수 있게 한다
            torch.save({**ckpt, "optimizer": opt.state_dict(), "best_em": best_em, "best_val_loss": best_val_loss,
                        "bad_epochs": bad_epochs, "history": history, "tokens_seen": tokens_seen,
                        "train_time": train_time,
                        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                                "syn": syn_rng.getstate() if syn_swapper else None,
                                "swap": swap_rng.getstate() if swapper else None}},
                       run_dir / "last.pt")
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
            ev = evaluate_set(model, name, tok, device, amp_dtype, eval_dir=args.eval_dir, db_path=args.db, rules=rules)
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
