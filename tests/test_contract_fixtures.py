"""GGUF 계약 fixture 가드: 계산 결과가 바뀌었는데 CONTRACT_VERSION 을 올리지 않으면 실패한다.

C 엔진(https://github.com/polring/capstone-sql-cli)은 이 fixture 의 사본으로만 검증되므로, Python 쪽 계산이
말없이 바뀌면 C 엔진과 어긋난 채로 배포된다. 의도한 변경이면 CONTRACT_VERSION 을 올리고
`python -m src.export_fixtures --new-model` 로 새 버전 fixture 를 만든다.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src import export_fixtures as fx
from src.exporter import export_model, read_gguf
from src.tokenizer.contract import CONTRACT_VERSION

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / fx.FIXTURE_ROOT / f"v{CONTRACT_VERSION}"
BUMP_HINT = (
    "계산 결과가 커밋된 fixture 와 다르다. 의도한 변경이면 src/tokenizer/contract.py 의"
    " CONTRACT_VERSION 을 올리고 `python -m src.export_fixtures --new-model` 을 실행한다"
)
SETS = {fixture.name: fixture for fixture in fx.FIXTURE_SETS}


def test_manifest_matches_contract_version_and_files():
    """manifest 의 계약 버전이 현재 CONTRACT_VERSION 과 같고, 모든 세트의 해시가 커밋된 파일과 같다."""
    path = FIXTURE_DIR / fx.MANIFEST_FILE
    assert path.exists(), (
        f"{FIXTURE_DIR} 가 없다. CONTRACT_VERSION 을 올렸다면"
        " `python -m src.export_fixtures --new-model` 로 fixture 를 만든다"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["contract_version"] == CONTRACT_VERSION
    expected_files = {
        f"{name}/{file}" for name in SETS for file in (fx.MODEL_FILE, fx.CASES_FILE)
    }
    assert set(manifest["files"]) == expected_files
    for name, digest in manifest["files"].items():
        assert fx.sha256(FIXTURE_DIR / name) == digest, f"{name} 가 manifest 와 다르다"


@pytest.mark.parametrize("name", SETS)
def test_fixture_model_reexports_identically(name, tmp_path):
    """커밋된 model.gguf 를 읽어 다시 내보내면 바이트가 같다 (GGUF 배치·메타데이터 불변)."""
    path = FIXTURE_DIR / name / fx.MODEL_FILE
    model, tokenizer, metadata = read_gguf(path)
    assert metadata["capstone_sql.contract_version"] == CONTRACT_VERSION
    output = tmp_path / "model.gguf"
    export_model(model, output, tokenizer.merges, tokenizer.config)
    assert output.read_bytes() == path.read_bytes(), BUMP_HINT


@pytest.mark.parametrize("name", SETS)
def test_recomputed_cases_match_committed(name):
    """커밋된 모델로 다시 계산한 토큰화·logits·생성 결과가 커밋된 기대값과 같다."""
    folder = FIXTURE_DIR / name
    committed = fx.parse_cases((folder / fx.CASES_FILE).read_text(encoding="utf-8"))
    recomputed = fx.compute_cases(SETS[name], folder / fx.MODEL_FILE)
    assert [case[:2] for case in recomputed] == [
        case[:2] for case in committed
    ], "fixture 케이스 목록이 바뀌었다. `python -m src.export_fixtures` 로 다시 만든다"
    for actual, expected in zip(recomputed, committed):
        if actual[0] == "logits":
            np.testing.assert_allclose(
                actual[2], expected[2], atol=2e-5, rtol=2e-5, err_msg=BUMP_HINT
            )
        else:
            assert actual == expected, BUMP_HINT
