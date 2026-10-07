"""
loading.py — 저장된 모델을 쓸 수 있게 불러온다

학습 체크포인트(runs/.../best.pt)와 배포 모델(models/stage<N>/model.pt)을 안전하게(weights_only) 읽고, 그 모델이
학습에 쓴 토크나이저를 찾는다. run.py, release.py, demo.py 가 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src import config
from src.models.model import TextToSQLModel


def load_model(ckpt_path: str | Path, device) -> tuple[TextToSQLModel, dict]:
    """학습 체크포인트(best.pt)와 배포 모델(model.pt) 모두 읽는다. 둘 다 텐서와 기본 자료형만 담고 있어
    weights_only=True 로 읽는다 (pickle 로 임의 코드가 실행되지 않게)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = TextToSQLModel(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def tokenizer_path_for(ckpt_path: str | Path, ckpt: dict) -> Path:
    """체크포인트를 학습할 때 쓴 토크나이저. 체크포인트 폴더에 사본(train.py 가 저장)이 있으면 그것을 쓴다 —
    코퍼스나 seed 목록이 바뀌어 config.TOKENIZER_PATH 가 다시 만들어져도 이전 체크포인트를 그대로 평가하기 위해서다.
    예전 실행 폴더에는 병합 규칙만 있는 bpe_merges.json 이 있고, Tokenizer.load 가 그 형식도 읽는다."""
    folder = Path(ckpt_path).parent
    for name in ("tokenizer.json", "bpe_merges.json"):
        if (folder / name).exists():
            return folder / name
    return Path(ckpt.get("tokenizer_path") or ckpt.get("merges_path") or config.TOKENIZER_PATH)


def describe_checkpoint(ckpt: dict) -> str:
    """출력용 요약: 학습 체크포인트면 epoch·검증 EM, 배포 모델(model.pt)이면 단계."""
    if "epoch" in ckpt:
        return f"epoch {ckpt['epoch']}, val EM {ckpt.get('val_em', 0):.4f}"
    return f"배포 모델, stage {ckpt.get('stage')}"
