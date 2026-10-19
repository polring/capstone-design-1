import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import PAD_ID, TextToSQLDataset, collate_fn, tokenize_pairs
from src.tokenizer import bpe
from src.models.model import TextToSQLModel
from src.train import evaluate_teacher_forced, lr_at, make_optimizer, masked_loss, parse_round_repeat, repeat_llm_pairs


def tiny_model() -> TextToSQLModel:
    torch.manual_seed(0)
    return TextToSQLModel(vocab_size=300, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                          max_seq_len=128, padding_idx=PAD_ID)


@pytest.mark.integration
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("mode, expected_repeats", [
    ("default", {"numbers": 1}),
    ("override", {"numbers": 2}),
    ("explicit", {}),
])
def test_training_keeps_canonical_split_and_recipe(tmp_path, monkeypatch, mode, expected_repeats, compressed):
    import gzip
    import json
    from types import SimpleNamespace
    from src import train

    # Import stage rules before using an isolated working directory.
    train.load_rules(3)
    folder = tmp_path / "data/stage3"
    training = [{"question": "train", "sql": "SELECT * FROM items", "source": "llm:gpt", "round": "numbers"}]
    validation = [{"question": "val", "sql": "SELECT * FROM customers"}]
    train_name = "train/pairs.json.gz" if compressed else "train/pairs.json"
    for name, value in ((train_name, training), ("validation/pairs.json", validation),
                        ("dataset_info.json", {"training_defaults": {"round_repeats": {"numbers": 1}}})):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as stream:
            json.dump(value, stream)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train.bpe.Tokenizer, "load", lambda _: SimpleNamespace(vocab_size=300))
    original_validate = train.validate_split
    validated = []

    def check_split(actual_train, actual_val):
        assert actual_train == training and actual_val == validation
        original_validate(actual_train, actual_val)
        validated.append(True)

    class DataReady(Exception):
        pass

    def stop_before_training(pairs, repeat, round_repeats):
        assert validated and pairs == training
        assert repeat == 3 and round_repeats == expected_repeats
        raise DataReady

    monkeypatch.setattr(train, "validate_split", check_split)
    monkeypatch.setattr(train, "repeat_llm_pairs", stop_before_training)
    args = ["--stage", "3", "--cpu", "--name-swap-ratio", "0"]
    if mode == "override":
        args += ["--llm-round-repeat", "numbers=2"]
    if mode == "explicit":
        args += ["--train-files", "data/stage3/" + train_name, "--val-files", "data/stage3/validation/pairs.json"]
    with pytest.raises(DataReady):
        train.main(args)


@pytest.mark.unit
def test_round_repeat_preserves_other_rounds_and_original_order():
    """numbers만 1회로 낮춰도 다른 LLM 3회·조합기 1회·기존 사본 순서를 보존한다"""
    base = {"question": "base", "sql": "S0", "source": "composer"}
    numbers = {"question": "numbers", "sql": "S1", "source": "llm:gpt", "round": "numbers"}
    intent = {"question": "intent", "sql": "S2", "source": "llm:gpt", "round": "intent"}
    pairs = [base, numbers, intent]
    assert repeat_llm_pairs(pairs, 3) == pairs + [numbers, intent] * 2
    assert repeat_llm_pairs(pairs, 3, {"numbers": 1}) == pairs + [intent, intent]
    assert pairs == [base, numbers, intent]
    assert repeat_llm_pairs(pairs, 1, {"numbers": 2}) == pairs + [numbers]


@pytest.mark.unit
def test_round_repeat_rejects_unknown_round_instead_of_ignoring_typo():
    """검증에만 있거나 이름을 잘못 쓴 라운드의 반복 설정을 조용히 무시하지 않는다"""
    pair = {"question": "q", "sql": "S", "source": "llm:gpt", "round": "numbers"}
    with pytest.raises(ValueError, match="absent from training"):
        repeat_llm_pairs([pair], 3, {"number": 1})
    with pytest.raises(ValueError, match="positive"):
        repeat_llm_pairs([pair], 3, {"numbers": 0})


@pytest.mark.unit
@pytest.mark.parametrize("value", ["numbers=0", "numbers=-1", "numbers=x", "numbers", "=1"])
def test_round_repeat_cli_requires_positive_count(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_round_repeat(value)


@pytest.mark.unit
def test_round_repeat_cli_parses_explicit_round():
    assert parse_round_repeat("llm_numbers_r6_600=1") == ("llm_numbers_r6_600", 1)


@pytest.mark.unit
@pytest.mark.parametrize("options", [["--steps-per-epoch", "0"], ["--steps-per-epoch", "-1"], ["--steps-per-epoch", "1", "--overfit", "1"]])
def test_step_cap_rejects_invalid_or_overfit_mode(options):
    from src import train
    with pytest.raises(SystemExit) as exc:
        train.main(["--stage", "3", *options])
    assert exc.value.code == 2


@pytest.mark.unit
@pytest.mark.parametrize("options", [
    ["--stage", "2", "--like-sampling", "balanced"],
    ["--stage", "3", "--like-sampling", "balanced", "--overfit", "1"],
    ["--stage", "3", "--like-sampling", "balanced", "--name-swap-ratio", "0"],
    ["--stage", "3", "--like-sampling", "balanced", "--name-swap-ratio", "-0.1"],
    ["--stage", "3", "--like-sampling", "balanced", "--name-swap-ratio", "1.1"],
])
def test_balanced_like_rejects_inactive_or_unsupported_training_modes(monkeypatch, options):
    """균형 LIKE 옵션을 다른 단계·과적합·교체 비활성 실행에서 조용히 무시하지 않는다"""
    from src import train
    monkeypatch.setattr(train, "load_train_pairs", lambda *args, **kwargs: [])
    with pytest.raises(SystemExit) as exc:
        train.main(options)
    assert exc.value.code == 2


# ===========================================================================
# masked_loss
# ===========================================================================

@pytest.mark.unit
def test_masked_loss_only_counts_masked_positions():
    """loss 는 loss_mask 위치의 평균 CE 이고, 마스크 밖 logits 를 바꿔도 변하지 않아야 함"""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    target = torch.randint(0, 11, (2, 5))
    mask = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    loss, _ = masked_loss(logits, target, mask)
    assert loss.item() == pytest.approx(F.cross_entropy(logits[mask], target[mask]).item(), rel=1e-6)

    noisy = logits.clone()
    noisy[~mask] += torch.randn_like(noisy[~mask]) * 100
    assert masked_loss(noisy, target, mask)[0].item() == pytest.approx(loss.item(), rel=1e-6)


@pytest.mark.unit
def test_masked_loss_correct_count():
    """맞힌 토큰 수는 마스크 안에서만 센다"""
    target = torch.tensor([[1, 2, 3]])
    logits = F.one_hot(torch.tensor([[1, 2, 0]]), 5).float()  # 위치 0, 1 정답 / 2 오답
    mask = torch.tensor([[False, True, True]])
    _, correct = masked_loss(logits, target, mask)
    assert correct.item() == 1


# ===========================================================================
# lr 스케줄 / optimizer
# ===========================================================================

@pytest.mark.unit
def test_lr_schedule_warmup_and_cosine():
    """선형 warmup → max_lr, 이후 cosine 으로 max_lr*min_ratio 까지 단조 감소"""
    max_lr, warmup, total = 1e-3, 10, 110
    assert lr_at(0, max_lr, warmup, total) == pytest.approx(max_lr / warmup)
    assert lr_at(warmup - 1, max_lr, warmup, total) == pytest.approx(max_lr)
    assert lr_at(warmup, max_lr, warmup, total) == pytest.approx(max_lr)
    assert lr_at(60, max_lr, warmup, total) == pytest.approx(max_lr * 0.55)  # 중간 지점: (0.1 + 1) / 2
    assert lr_at(total, max_lr, warmup, total) == pytest.approx(max_lr * 0.1)
    assert lr_at(total * 2, max_lr, warmup, total) == pytest.approx(max_lr * 0.1)
    decay = [lr_at(s, max_lr, warmup, total) for s in range(warmup, total + 1)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))


@pytest.mark.unit
def test_make_optimizer_weight_decay_groups():
    """2D 이상 파라미터에만 weight decay, RMSNorm weight(1D)는 0, 모든 파라미터가 한 번씩 포함"""
    model = tiny_model()
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"]) and no_decay["params"]
    grouped = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())


# ===========================================================================
# Integration
# ===========================================================================

@pytest.mark.integration
def test_evaluate_teacher_forced_and_train_step():
    """teacher-forced 평가가 유한한 loss·[0,1] 정확도를 내고, 몇 step 학습하면 loss 가 줄어드는지 검증"""
    pairs = [{"question": f"item {i}", "sql": f"SELECT * FROM items WHERE item_id = {i}"} for i in range(8)]
    loader = DataLoader(TextToSQLDataset(tokenize_pairs(pairs, bpe.Tokenizer(bpe.load_seed_tokens()))), batch_size=8, collate_fn=collate_fn)
    model = tiny_model()
    device = torch.device("cpu")

    loss0, acc0 = evaluate_teacher_forced(model, loader, device, None)
    assert torch.isfinite(torch.tensor(loss0)) and 0.0 <= acc0 <= 1.0
    assert model.training, "evaluate_teacher_forced 후 train 모드로 돌아가야 함"

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = next(iter(loader))
    for _ in range(30):
        loss, _ = masked_loss(model(batch["input_ids"]), batch["target_ids"], batch["loss_mask"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss1, _ = evaluate_teacher_forced(model, loader, device, None)
    assert loss1 < loss0
