import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import PAD_ID, TextToSQLDataset, collate_fn, tokenize_pairs
from src.models.model import TextToSQLModel
from src.train import evaluate_teacher_forced, lr_at, make_optimizer, masked_loss, split_by_sql


def make_pairs(n_sql: int = 20, per_sql: int = 5) -> list[dict]:
    return [{"question": f"q{i}-{j}", "sql": f"SELECT * FROM items WHERE item_id = {3000 + i}"}
            for i in range(n_sql) for j in range(per_sql)]


def tiny_model() -> TextToSQLModel:
    torch.manual_seed(0)
    return TextToSQLModel(vocab_size=300, dim=32, num_layers=1, num_heads=2, ffn_dim=64,
                          max_seq_len=128, padding_idx=PAD_ID)


# ===========================================================================
# split_by_sql
# ===========================================================================

@pytest.mark.unit
def test_split_by_sql_no_sql_overlap():
    """같은 SQL 의 질문들은 train / val 한쪽에만 들어가야 함 (val 이 암기 점수가 되지 않게)"""
    pairs = make_pairs()
    train, val = split_by_sql(pairs, val_ratio=0.2, seed=0)
    assert not {p["sql"] for p in train} & {p["sql"] for p in val}
    assert len({p["sql"] for p in val}) == 4  # 20 SQL * 0.2
    assert len(train) + len(val) == len(pairs)


@pytest.mark.unit
def test_split_by_sql_deterministic():
    """같은 seed 면 같은 분할, 다른 seed 면 다른 분할"""
    pairs = make_pairs()
    assert split_by_sql(pairs, 0.2, 0) == split_by_sql(pairs, 0.2, 0)
    assert split_by_sql(pairs, 0.2, 0)[1] != split_by_sql(pairs, 0.2, 1)[1]


@pytest.mark.unit
def test_split_by_sql_uses_original_pair_for_swapped_data():
    """이름 교체 쌍은 orig_sql 로 나누고, val 에는 원래 질문·SQL 이 들어가야 함 (기준 실행과 같은 val)"""
    pairs = make_pairs()
    swapped = [dict(p, question=p["question"] + " zorb", sql=p["sql"] + "9",
                    orig_question=p["question"], orig_sql=p["sql"]) if i % 2 else p
               for i, p in enumerate(pairs)]
    _, val_plain = split_by_sql(pairs, 0.2, 0)
    train_sw, val_sw = split_by_sql(swapped, 0.2, 0)
    assert val_sw == [{"question": p["question"], "sql": p["sql"]} for p in val_plain]
    assert all(p.get("orig_sql", p["sql"]) not in {v["sql"] for v in val_sw} for p in train_sw)


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
    loader = DataLoader(TextToSQLDataset(tokenize_pairs(pairs, [])), batch_size=8, collate_fn=collate_fn)
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
