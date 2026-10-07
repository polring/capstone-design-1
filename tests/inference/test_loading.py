import json
from pathlib import Path

import pytest
import torch

from src.inference import loading
from src.data.dataset import PAD_ID
from src.models.model import TextToSQLModel


VOCAB = 300


@pytest.mark.unit
def test_tokenizer_path_prefers_run_copy(tmp_path):
    """체크포인트 폴더의 tokenizer.json > 예전 bpe_merges.json > ckpt 에 적힌 경로 순으로 쓴다"""
    ckpt_path = tmp_path / "best.pt"
    assert loading.tokenizer_path_for(ckpt_path, {"tokenizer_path": "x.json"}) == Path("x.json")
    assert loading.tokenizer_path_for(ckpt_path, {"merges_path": "old.json"}) == Path("old.json")
    (tmp_path / "bpe_merges.json").write_text("[]", encoding="utf-8")
    assert loading.tokenizer_path_for(ckpt_path, {}) == tmp_path / "bpe_merges.json"
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert loading.tokenizer_path_for(ckpt_path, {}) == tmp_path / "tokenizer.json"


@pytest.mark.integration
def test_load_model_roundtrip(tmp_path):
    """train.py 형식 체크포인트(model + model_cfg)를 같은 출력의 모델로 다시 불러오는지 검증"""
    cfg = dict(vocab_size=VOCAB, dim=32, num_layers=1, num_heads=2, ffn_dim=64, max_seq_len=48, padding_idx=PAD_ID)
    torch.manual_seed(0)
    model = TextToSQLModel(**cfg).eval()
    torch.save({"model": model.state_dict(), "model_cfg": cfg}, tmp_path / "best.pt")
    loaded, ckpt = loading.load_model(tmp_path / "best.pt", torch.device("cpu"))
    idx = torch.randint(0, 256, (1, 8))
    assert not loaded.training
    assert torch.allclose(model(idx), loaded(idx))
