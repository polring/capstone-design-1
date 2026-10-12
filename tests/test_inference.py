"""GGUF export tests: the file written for the C engine holds the same model.

The C engine itself lives in https://github.com/polring/capstone-sql-cli and is
checked there against tests/fixtures/contract/ (see test_contract_fixtures.py).
Here every exported file is read back with exporter.read_gguf and compared with
the PyTorch model it came from.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.dataset import BOS_ID, EOS_ID, SEP_ID
from src.export_fixtures import reference_generate
from src.exporter import export_model, main as exporter_main, read_gguf
from src.inference.gguf_model import InferenceModel, ModelConfig, load_checkpoint
from src.inference.loading import load_model, tokenizer_path_for
from src.models.model import TextToSQLModel
from src.tokenizer import bpe

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


def last_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids]))[0, -1].numpy()


@pytest.fixture
def fixture_model(tmp_path):
    torch.manual_seed(17)
    config = ModelConfig(
        vocab_size=320, dim=16, heads=2, layers=2, ffn_dim=24, context=128
    )
    model = InferenceModel(config).eval()
    seeds = bpe.load_seed_tokens()
    merges = bpe.Tokenizer.train(
        ["hello hello", "stock mouse", "SELECT stock FROM items;"],
        seeds,
        vocab_size=bpe.SEED_BASE + len(seeds) + 12,
    ).merges
    path = tmp_path / "model.gguf"
    export_model(model, path, merges)
    return model, path, merges


def test_round_trip_preserves_weights_tokenizer_and_logits(fixture_model):
    """내보낸 GGUF 를 다시 읽으면 가중치·병합 규칙·logits 가 원래 모델과 같다."""
    model, path, merges = fixture_model
    loaded, tokenizer, _ = read_gguf(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert tokenizer.merges == merges
    assert tokenizer.seed_tokens == bpe.load_seed_tokens()
    ids = [257, 104, 105, 259]
    np.testing.assert_allclose(
        last_logits(loaded, ids), last_logits(model, ids), atol=1e-6, rtol=1e-6
    )


def test_config_variation_round_trip(tmp_path):
    """eps·theta 등 기본값이 아닌 설정도 메타데이터로 전달된다."""
    torch.manual_seed(5)
    config = ModelConfig(
        vocab_size=300,
        dim=8,
        heads=1,
        layers=1,
        ffn_dim=1,
        context=16,
        eps=1e-5,
        theta=5000,
    )
    model = InferenceModel(config).eval()
    path = tmp_path / "variant.gguf"
    export_model(model, path, [])
    loaded, _, metadata = read_gguf(path)
    assert metadata["capstone_sql.theta"] == 5000
    assert metadata["capstone_sql.eps"] == pytest.approx(1e-5)
    assert loaded.config.context == 16 and loaded.config.ffn_dim == 1
    ids = [257, 1, 259]
    np.testing.assert_allclose(
        last_logits(loaded, ids), last_logits(model, ids), atol=1e-6, rtol=1e-6
    )


def test_independent_gguf_reader(fixture_model):
    """독립 GGUF 구현(gguf 패키지)으로도 모든 텐서 값이 그대로 읽힌다."""
    gguf = pytest.importorskip("gguf")
    model, path, _ = fixture_model
    reader = gguf.GGUFReader(path)
    assert len(reader.tensors) == len(model.state_dict())
    for tensor in reader.tensors:
        expected = model.state_dict()[tensor.name].detach().numpy()
        np.testing.assert_array_equal(tensor.data.reshape(expected.shape), expected)


def test_checkpoint_contract(fixture_model, tmp_path):
    """config + state_dict 체크포인트를 그대로 불러오고, 잘못된 병합 규칙은 내보내지 않는다."""
    model, _, merges = fixture_model
    checkpoint = tmp_path / "model.pt"
    torch.save(model.checkpoint(), checkpoint)
    loaded = load_checkpoint(checkpoint)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    with pytest.raises(ValueError, match="merges"):
        export_model(loaded, tmp_path / "bad.gguf", [((99999, 1), 298)])


def test_exporter_converts_training_checkpoint(tmp_path, capsys):
    """배포 형식 체크포인트는 --tokenizer 없이 내보내지고, 토큰화·logits 가 학습 모델과 같다."""
    torch.manual_seed(5)
    seeds = bpe.load_seed_tokens()
    tokenizer = bpe.Tokenizer.train(
        ["hello hello", "stock mouse", "SELECT stock FROM items;"],
        seeds,
        vocab_size=bpe.SEED_BASE + len(seeds) + 12,
    )
    cfg = {
        "vocab_size": 320,
        "dim": 16,
        "num_layers": 2,
        "num_heads": 2,
        "ffn_dim": 43,
        "max_seq_len": 64,
        "padding_idx": 256,
    }
    team = TextToSQLModel(**cfg).eval()
    release = tmp_path / "release"
    release.mkdir()
    torch.save(
        {"format_version": 1, "stage": 1, "model_cfg": cfg, "model": team.state_dict()},
        release / "model.pt",
    )
    tokenizer.save(release / "tokenizer.json")
    path = tmp_path / "model.gguf"
    exporter_main(["--checkpoint", str(release / "model.pt"), "--output", str(path)])
    capsys.readouterr()

    loaded, gguf_tokenizer, _ = read_gguf(path)
    text = "hello stock"
    assert gguf_tokenizer.encode(text) == tokenizer.encode(text)
    ids = [BOS_ID] + tokenizer.encode(text) + [SEP_ID]
    np.testing.assert_allclose(
        last_logits(loaded, ids), last_logits(team, ids), atol=2e-5, rtol=2e-5
    )


def test_exporter_requires_tokenizer_for_contract_checkpoint(fixture_model, tmp_path):
    """토크나이저 위치가 없는 config + state_dict 체크포인트는 --tokenizer 가 필요하다."""
    model, _, _ = fixture_model
    checkpoint = tmp_path / "model.pt"
    torch.save(model.checkpoint(), checkpoint)
    with pytest.raises(SystemExit):
        exporter_main(
            ["--checkpoint", str(checkpoint), "--output", str(tmp_path / "x.gguf")]
        )


@pytest.mark.parametrize(
    "question",
    [
        "what city does ashley live in?",
        "how many mouse units are left in stock?",
        "show me everything about order 5012",
    ],
)
def test_release_model_generates_same_sql_after_export(release_gguf, question):
    """배포 모델을 내보낸 GGUF 로 계약 규칙대로 생성하면 학습 모델과 같은 SQL 이 나온다."""
    team, tokenizer, path = release_gguf
    prompt = [BOS_ID] + tokenizer.encode(question) + [SEP_ID]
    expected = tokenizer.decode(team.generate(prompt, EOS_ID, max_new_tokens=64))
    loaded, gguf_tokenizer, _ = read_gguf(path)
    output, stop = reference_generate(loaded, gguf_tokenizer, question, 64)
    assert expected.startswith("SELECT ")
    assert output.decode("utf-8") == expected
    assert stop == "eos"


@pytest.fixture(scope="module")
def release_gguf(tmp_path_factory):
    checkpoint = ROOT / "models/stage1/model.pt"
    if not checkpoint.exists():
        pytest.skip("release model is not available")
    team, raw = load_model(checkpoint, torch.device("cpu"))
    tokenizer = bpe.Tokenizer.load(tokenizer_path_for(checkpoint, raw))
    path = tmp_path_factory.mktemp("release") / "model.gguf"
    exporter_main(["--checkpoint", str(checkpoint), "--output", str(path)])
    return team, tokenizer, path
