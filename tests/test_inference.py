"""End-to-end tests: real team decoder vs compiled C engine, not two toy mocks."""

import shutil
import struct
import subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from src.exporter import export_model
from src.inference import InferenceModel, ModelConfig, load_checkpoint
from src.tokenizer import bpe

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    if not shutil.which("gcc"):
        pytest.skip("GCC is required for standalone C integration tests")
    output = tmp_path_factory.mktemp("engine") / "loader.exe"
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(ROOT / "src/loader.c"),
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def run(engine, path, *args, check=True):
    return subprocess.run(
        [str(engine), str(path), *map(str, args)], capture_output=True, check=check
    )


@pytest.fixture
def fixture_model(tmp_path):
    torch.manual_seed(17)
    config = ModelConfig(
        vocab_size=320, dim=16, heads=2, layers=2, ffn_dim=24, context=128
    )
    model = InferenceModel(config).eval()
    merges, _ = bpe.train(
        ["hello hello", "stock mouse", "SELECT stock FROM items;"], merge_budget=12
    )
    path = tmp_path / "model.gguf"
    export_model(model, path, merges)
    return model, path, merges


@pytest.mark.parametrize(
    "ids", [[257], [257, 104, 105, 259], [257, 1, 2, 3, 4, 5, 259]]
)
def test_python_c_logits(engine, fixture_model, ids):
    model, path, _ = fixture_model
    with torch.no_grad():
        expected = model(torch.tensor([ids]))[0, -1].numpy()
    actual = np.fromstring(run(engine, path, "--logits", *ids).stdout.decode(), sep=" ")
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual.argmax() == expected.argmax()


def test_config_variation(engine, tmp_path):
    torch.manual_seed(5)
    model = InferenceModel(
        ModelConfig(
            vocab_size=300,
            dim=8,
            heads=1,
            layers=1,
            ffn_dim=1,
            context=16,
            eps=1e-5,
            theta=5000,
        )
    ).eval()
    path = tmp_path / "variant.gguf"
    export_model(model, path, [])
    ids = [257, 1, 259]
    with torch.no_grad():
        expected = model(torch.tensor([ids]))[0, -1].numpy()
    actual = np.fromstring(run(engine, path, "--logits", *ids).stdout.decode(), sep=" ")
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_independent_gguf_reader(fixture_model):
    gguf = pytest.importorskip("gguf")
    model, path, _ = fixture_model
    reader = gguf.GGUFReader(path)
    assert len(reader.tensors) == len(model.state_dict())
    for tensor in reader.tensors:
        expected = model.state_dict()[tensor.name].detach().numpy()
        np.testing.assert_array_equal(tensor.data.reshape(expected.shape), expected)


@pytest.mark.parametrize(
    "text",
    [
        "hello hello",
        "SELECT stock FROM items;",
        "new york_ new york",
        "customers.name = 'mouse'",
        "123 45.6\n",
        "stockholm stock",
    ],
)
def test_team_bpe_matches_c(engine, fixture_model, text):
    _, path, merges = fixture_model
    actual = list(map(int, run(engine, path, "--encode", text).stdout.split()))
    assert actual == bpe.encode(text, merges)


def test_generation_repeats_python_forward(engine, fixture_model):
    model, path, merges = fixture_model
    question = "hello"
    ids = [257] + bpe.encode(question, merges) + [259]
    pieces = bpe.id_to_bytes_from_merges(merges)
    expected = bytearray()
    for _ in range(5):
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        allowed = sorted(set(pieces) | {258})
        next_id = max(allowed, key=lambda i: float(logits[i]))
        if next_id == 258:
            break
        expected.extend(pieces[next_id])
        ids.append(next_id)
    result = run(engine, path, "--generate", question, 5)
    assert result.stdout == expected + b"\n"
    assert b"stop=" in result.stderr


def test_eos_and_context_stop(engine, fixture_model):
    model, path, merges = fixture_model
    # Zero all logits: C's deterministic tie policy picks EOS immediately.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    export_model(model, path, merges)
    result = run(engine, path, "--generate", "hello", 8)
    assert result.stdout == b"\n" and b"stop=eos" in result.stderr
    # Prompt exactly fills context: no overflow or extra forward call.
    result = run(engine, path, "--generate", "1" * 126, 8)
    assert b"stop=context" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--logits",),
        ("--logits", "320"),
        ("--logits", "-1"),
        ("--encode", "한글"),
        ("--generate", "1" * 127, "1"),
    ],
)
def test_bad_inputs_rejected(engine, fixture_model, args):
    _, path, _ = fixture_model
    result = run(engine, path, *args, check=False)
    assert result.returncode != 0 and b"error:" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    ["truncated", "version", "architecture", "tensor_type", "shape", "offset"],
)
def test_bad_files_rejected(engine, fixture_model, mutation, tmp_path):
    _, path, _ = fixture_model
    raw = bytearray(path.read_bytes())
    if mutation == "truncated":
        raw = raw[:-64]
    elif mutation == "version":
        struct.pack_into("<I", raw, 4, 2)
    elif mutation == "architecture":
        pos = raw.index(b"capstone_sql", raw.index(b"general.architecture") + 20)
        raw[pos : pos + 12] = b"other_model!"
    else:
        # Locate the first descriptor independently through its name bytes.
        pos = raw.index(b"embedding.embedding.weight") + len(
            b"embedding.embedding.weight"
        )
        rank = struct.unpack_from("<I", raw, pos)[0]
        if mutation == "tensor_type":
            struct.pack_into("<I", raw, pos + 4 + rank * 8, 1)
        elif mutation == "shape":
            struct.pack_into("<Q", raw, pos + 4, 15)
        else:
            struct.pack_into("<Q", raw, pos + 8 + rank * 8, 2**63)
    invalid = tmp_path / "invalid.gguf"
    invalid.write_bytes(raw)
    result = run(engine, invalid, "--logits", 257, check=False)
    assert result.returncode != 0 and b"error:" in result.stderr


def test_checkpoint_contract(fixture_model, tmp_path):
    model, _, merges = fixture_model
    checkpoint = tmp_path / "model.pt"
    torch.save(model.checkpoint(), checkpoint)
    loaded = load_checkpoint(checkpoint)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    with pytest.raises(ValueError, match="merges"):
        export_model(loaded, tmp_path / "bad.gguf", [((99999, 1), 298)])


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("heads", 0, b"invalid dimension"),
        ("vocab_size", 255, b"incomplete or incompatible config"),
        ("dim", 15, b"incomplete or incompatible config"),
        ("heads", 16, b"incomplete or incompatible config"),
        ("layers", 257, b"incomplete or incompatible config"),
        ("context", 4097, b"incomplete or incompatible config"),
    ],
)
def test_c_rejects_incompatible_config(
    engine, fixture_model, tmp_path, field, value, message
):
    _, path, _ = fixture_model
    raw = bytearray(path.read_bytes())
    key = f"capstone_sql.{field}".encode()
    # A scalar value follows the key bytes and its four-byte GGUF type.
    value_position = raw.index(key) + len(key) + 4
    struct.pack_into("<I", raw, value_position, value)
    invalid = tmp_path / "bad_config.gguf"
    invalid.write_bytes(raw)
    result = run(engine, invalid, "--logits", 257, check=False)
    assert result.returncode != 0 and message in result.stderr


def test_c_rejects_overlapping_tensor_data(engine, fixture_model, tmp_path):
    _, path, _ = fixture_model
    raw = bytearray(path.read_bytes())
    name = b"blocks.0.attn_norm.weight"
    descriptor_position = raw.index(name) + len(name)
    rank = struct.unpack_from("<I", raw, descriptor_position)[0]
    offset_position = descriptor_position + 8 + rank * 8
    # The first tensor already starts at offset zero; overlap the next tensor.
    struct.pack_into("<Q", raw, offset_position, 0)
    invalid = tmp_path / "overlap.gguf"
    invalid.write_bytes(raw)
    result = run(engine, invalid, "--logits", 257, check=False)
    assert result.returncode != 0 and b"overlapping tensor data" in result.stderr


@pytest.mark.parametrize("input_id", [256, 298, 300])
def test_c_rejects_missing_or_forward_merge_references(
    engine, fixture_model, tmp_path, input_id
):
    _, path, _ = fixture_model
    raw = bytearray(path.read_bytes())
    key = b"capstone_sql.tokenizer.merges"
    # Skip the array type, element type and element count to reach merge[0].a.
    first_merge_position = raw.index(key) + len(key) + 16
    struct.pack_into("<I", raw, first_merge_position, input_id)
    invalid = tmp_path / "bad_merge.gguf"
    invalid.write_bytes(raw)
    result = run(engine, invalid, "--logits", 257, check=False)
    assert result.returncode != 0 and b"invalid merge references" in result.stderr


@pytest.mark.parametrize("value", ["2x", "2147483648", "999999999999999999999999"])
def test_c_rejects_invalid_integer_arguments(engine, fixture_model, value):
    _, path, _ = fixture_model
    result = run(engine, path, "--logits", value, check=False)
    assert result.returncode != 0 and b"expected nonnegative integer" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("--encode",),
        ("--encode", "hello", "extra"),
        ("--generate",),
        ("--generate", "hello", "1", "extra"),
    ],
)
def test_c_rejects_invalid_mode_argument_counts(engine, fixture_model, args):
    _, path, _ = fixture_model
    result = run(engine, path, *args, check=False)
    assert (
        result.returncode != 0 and b"unknown mode or invalid arguments" in result.stderr
    )


@pytest.mark.parametrize(
    "text",
    ["", " ", "\t\n", "stock stock", "stock_ stockholm", "1stock", "_stock", "aaaaaa"],
)
def test_encode_handler_preserves_piece_boundaries(engine, fixture_model, text):
    _, path, merges = fixture_model
    result = run(engine, path, "--encode", text)
    assert list(map(int, result.stdout.split())) == bpe.encode(text, merges)


def test_generation_handler_normalizes_uppercase_questions(engine, fixture_model):
    _, path, _ = fixture_model
    lowercase = run(engine, path, "--generate", "hello stock", 3)
    uppercase = run(engine, path, "--generate", "HELLO STOCK", 3)
    assert uppercase.stdout == lowercase.stdout
    assert uppercase.stderr == lowercase.stderr


def test_generation_handler_zero_budget(engine, fixture_model):
    _, path, _ = fixture_model
    result = run(engine, path, "--generate", "", 0)
    assert result.stdout == b"\n"
    assert b"stop=max_new_tokens" in result.stderr


def test_generation_handler_default_budget_matches_explicit_budget(
    engine, fixture_model
):
    _, path, _ = fixture_model
    default = run(engine, path, "--generate", "hello")
    explicit = run(engine, path, "--generate", "hello", 32)
    assert default.stdout == explicit.stdout
    assert default.stderr == explicit.stderr


def test_generation_selection_skips_special_ids_and_breaks_ties(engine, fixture_model):
    model, path, merges = fixture_model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.embedding.embedding.weight.fill_(1)
        model.final_norm.weight.fill_(1)
        # The smaller decodable ID must win the tie between A and B.
        model.lm_head.weight[65].fill_(1)
        model.lm_head.weight[66].fill_(1)
        # These IDs have larger logits but cannot be emitted as token bytes.
        for token_id in (256, 257, 259, 319):
            model.lm_head.weight[token_id].fill_(2)
    export_model(model, path, merges)
    result = run(engine, path, "--generate", "", 2)
    assert result.stdout == b"AA\n"
    assert b"stop=max_new_tokens" in result.stderr


def test_missing_mode_prints_usage(engine, fixture_model):
    _, path, _ = fixture_model
    result = run(engine, path, check=False)
    assert result.returncode == 2
    assert result.stdout == b"" and b"usage:" in result.stderr


def test_cli_loads_model_before_rejecting_unknown_mode(engine, tmp_path):
    missing_model = tmp_path / "missing.gguf"
    result = run(engine, missing_model, "--unknown", check=False)
    assert result.returncode != 0
    assert b"cannot open model file" in result.stderr


def test_logits_handler_accepts_full_context_and_rejects_overflow(
    engine, fixture_model
):
    model, path, _ = fixture_model
    ids = [257] + [104] * (model.config.context - 1)
    with torch.no_grad():
        expected = model(torch.tensor([ids]))[0, -1].numpy()
    result = run(engine, path, "--logits", *ids)
    actual = np.fromstring(result.stdout.decode(), sep=" ")
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    overflow = run(engine, path, "--logits", *ids, 104, check=False)
    assert overflow.returncode != 0 and b"context length exceeded" in overflow.stderr
