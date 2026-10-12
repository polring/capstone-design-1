"""GGUF 계약 fixture 가드: 계산 결과가 바뀌었는데 CONTRACT_VERSION 을 올리지 않으면 실패한다.

C 엔진은 cli/fixtures/v<N>/ 의 기대값으로만 검증되므로(cli/CONTRACT.md), Python 쪽 계산이 말없이 바뀌면
C 엔진과 어긋난 채로 배포된다. 의도한 변경이면 CONTRACT_VERSION 을 올리고
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


@pytest.fixture(scope="module")
def manifest():
    path = FIXTURE_DIR / fx.MANIFEST_FILE
    assert path.exists(), (
        f"{FIXTURE_DIR} 가 없다. CONTRACT_VERSION 을 올렸다면"
        " `python -m src.export_fixtures --new-model` 로 fixture 를 만든다"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed_cases():
    text = (FIXTURE_DIR / fx.CASES_FILE).read_text(encoding="utf-8")
    return fx.parse_cases(text)


@pytest.fixture(scope="module")
def recomputed_cases():
    return fx.compute_cases(FIXTURE_DIR / fx.MODEL_FILE)


def test_manifest_matches_contract_version_and_files(manifest):
    """manifest 의 계약 버전이 현재 CONTRACT_VERSION 과 같고, 해시가 커밋된 파일과 같다."""
    assert manifest["contract_version"] == CONTRACT_VERSION
    for name, digest in manifest["files"].items():
        assert fx.sha256(FIXTURE_DIR / name) == digest, f"{name} 가 manifest 와 다르다"


def test_fixture_model_reexports_identically(tmp_path):
    """커밋된 model.gguf 를 읽어 다시 내보내면 바이트가 같다 (GGUF 배치·메타데이터 불변)."""
    path = FIXTURE_DIR / fx.MODEL_FILE
    model, tokenizer, metadata = read_gguf(path)
    assert metadata["capstone_sql.contract_version"] == CONTRACT_VERSION
    output = tmp_path / "model.gguf"
    export_model(model, output, tokenizer.merges)
    assert output.read_bytes() == path.read_bytes(), BUMP_HINT


def test_recomputed_cases_match_committed(committed_cases, recomputed_cases):
    """커밋된 모델로 다시 계산한 토큰화·logits·생성 결과가 커밋된 기대값과 같다."""
    assert [case[:2] for case in recomputed_cases] == [
        case[:2] for case in committed_cases
    ], "fixture 케이스 목록이 바뀌었다. `python -m src.export_fixtures` 로 다시 만든다"
    for actual, expected in zip(recomputed_cases, committed_cases):
        if actual[0] == "logits":
            np.testing.assert_allclose(
                actual[2], expected[2], atol=2e-5, rtol=2e-5, err_msg=BUMP_HINT
            )
        else:
            assert actual == expected, BUMP_HINT
