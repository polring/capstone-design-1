import json

import pytest
import torch

from src import config
from src import release as rl
from src.data.dataset import PAD_ID
from src.inference.loading import load_model
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

SEEDS = bpe.load_seed_tokens()


def _train_ckpt(tmp_path, vocab_size):
    """train.py 형식 체크포인트 + 같은 폴더의 tokenizer.json"""
    cfg = dict(vocab_size=vocab_size, dim=32, num_layers=1, num_heads=2, ffn_dim=64, max_seq_len=48, padding_idx=PAD_ID)
    torch.manual_seed(0)
    model = TextToSQLModel(**cfg).eval()
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"model": model.state_dict(), "model_cfg": cfg, "stage": 1, "epoch": 3, "step": 30,
                "val_em": 0.5, "val_loss": 1.0, "tokenizer_path": str(run / "tokenizer.json"),
                "args": {"seed": 7, "lr": 1e-3, "wordlist": "C:/Users/someone/wordnet.zip"}}, run / "best.pt")
    bpe.Tokenizer(SEEDS).save(run / "tokenizer.json")
    return model, run / "best.pt"


@pytest.mark.unit
def test_release_writes_self_contained_model_folder(tmp_path):
    """model.pt(가중치·설정만) + tokenizer.json + model_info.json 을 쓰고, 다시 불러온 모델 출력이 같아야 함"""
    model, ckpt = _train_ckpt(tmp_path, vocab_size=320)
    out = tmp_path / "models"
    m = rl.release(ckpt, out, run_eval=False)

    saved = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    assert set(saved) == {"format_version", "stage", "model_cfg", "model"}
    loaded, _ = load_model(out / "model.pt", torch.device("cpu"))
    idx = torch.randint(0, 256, (1, 8))
    assert torch.allclose(model(idx), loaded(idx))

    assert bpe.Tokenizer.load(out / "tokenizer.json").seed_tokens == SEEDS
    assert m["train_args"] == {"seed": 7, "lr": 1e-3}  # 경로가 든 인자는 남기지 않는다
    assert "someone" not in (out / "model_info.json").read_text(encoding="utf-8")
    assert "eval" not in m


@pytest.mark.unit
def test_release_rejects_tokenizer_larger_than_model(tmp_path):
    """토크나이저 vocab 이 모델 임베딩보다 크면 짝이 맞지 않는 것이므로 거부"""
    _, ckpt = _train_ckpt(tmp_path, vocab_size=280)
    with pytest.raises(ValueError):
        rl.release(ckpt, tmp_path / "models", run_eval=False)


@pytest.mark.unit
def test_committed_model_folder_is_consistent():
    """커밋된 배포 모델: weights_only 로 읽히고, 단계·토크나이저·model_info 가 서로 맞아야 함"""
    saved = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    tok = bpe.Tokenizer.load(config.MODEL_DIR / "tokenizer.json")
    info = json.loads((config.MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    assert saved["stage"] == config.STAGE == info["stage"]
    assert tok.vocab_size <= saved["model_cfg"]["vocab_size"]
    assert saved["model"]["tok_emb.embedding.weight"].shape[0] == saved["model_cfg"]["vocab_size"]
    assert info["tokenizer"]["vocab_size"] == tok.vocab_size
