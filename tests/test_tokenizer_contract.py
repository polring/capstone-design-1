"""Contract v2 must not depend on the stage-1 tokenizer's ID layout."""

from dataclasses import asdict, replace
import json
import struct

import pytest
import torch

from src.exporter import export_model
from src.inference import InferenceModel, ModelConfig
from src.tokenizer.contract import TokenizerConfig
from tests.test_inference import engine, run  # Reuse the compiled C fixture.


def custom_layout(seed_count=2):
    return TokenizerConfig(
        seeds=tuple(f"seed{index}" for index in range(seed_count)),
        pad_id=266,
        bos_id=267,
        eos_id=268,
        sep_id=269,
        seed_base=270,
        merge_base=270 + seed_count,
    )


def read_metadata(path):
    """Independent minimal reader; do not use writer helpers to check its bytes."""
    raw = path.read_bytes()
    magic, version, _, count = struct.unpack_from("<4sIQQ", raw)
    assert magic == b"GGUF" and version == 3
    position = 24

    def string():
        nonlocal position
        length = struct.unpack_from("<Q", raw, position)[0]
        position += 8
        value = raw[position : position + length].decode("utf-8")
        position += length
        return value

    fields = {}
    for _ in range(count):
        key = string()
        kind = struct.unpack_from("<I", raw, position)[0]
        position += 4
        if kind == 8:
            value = string()
        elif kind in (4, 6):
            value = struct.unpack_from("<I" if kind == 4 else "<f", raw, position)[0]
            position += 4
        else:
            assert kind == 9
            element_type, length = struct.unpack_from("<IQ", raw, position)
            position += 12
            if element_type == 8:
                value = [string() for _ in range(length)]
            else:
                assert element_type == 4
                value = list(struct.unpack_from(f"<{length}I", raw, position))
                position += 4 * length
        fields[key] = value
    return fields


@pytest.fixture
def custom_model(tmp_path):
    torch.manual_seed(23)
    model = InferenceModel(
        ModelConfig(vocab_size=320, dim=8, heads=1, layers=1, ffn_dim=12)
    ).eval()
    layout = custom_layout()
    merges = [((104, 105), layout.merge_base)]
    path = tmp_path / "custom.gguf"
    export_model(model, path, merges, layout)
    return model, path, layout, merges


@pytest.mark.integration
def test_export_serializes_tokenizer_layout(custom_model):
    _, path, layout, _ = custom_model
    fields = read_metadata(path)
    assert len(fields) == 19
    assert fields["capstone_sql.contract_version"] == 2
    assert fields["capstone_sql.tokenizer.seeds"] == list(layout.seeds)
    for key, value in asdict(layout).items():
        if key != "seeds":
            assert fields["capstone_sql.tokenizer." + key] == value
    assert fields["capstone_sql.tokenizer.merges"] == [104, 105, layout.merge_base]


@pytest.mark.unit
def test_tokenizer_config_json_roundtrip(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(asdict(custom_layout())), encoding="utf-8")
    assert TokenizerConfig.load(path) == custom_layout()


@pytest.mark.parametrize("seed_count", [0, 2, 45])
@pytest.mark.integration
def test_export_accepts_variable_seed_count(tmp_path, seed_count):
    layout = custom_layout(seed_count)
    model = InferenceModel(ModelConfig(vocab_size=320, dim=8, heads=1, layers=1))
    path = tmp_path / "variable.gguf"
    export_model(model, path, [], layout)
    assert len(read_metadata(path)["capstone_sql.tokenizer.seeds"]) == seed_count


@pytest.mark.parametrize(
    "changes",
    [
        {"pad_id": 0},
        {"bos_id": True},
        {"eos_id": 267},
        {"sep_id": 270},
        {"seed_base": 255},
        {"merge_base": 271},
        {"merge_base": 321},
        {"seeds": ("seed0", "seed0")},
        {"seeds": (" spaced",)},
        {"seeds": ("한글",)},
        {"seeds": ("nul\0seed",)},
    ],
)
@pytest.mark.integration
def test_export_rejects_invalid_layout_before_writing(custom_model, tmp_path, changes):
    model, _, layout, _ = custom_model
    path = tmp_path / "invalid.gguf"
    with pytest.raises(ValueError):
        export_model(model, path, [], replace(layout, **changes))
    assert not path.exists()


@pytest.mark.integration
def test_export_rejects_merge_using_special_token(custom_model, tmp_path):
    model, _, layout, _ = custom_model
    with pytest.raises(ValueError, match="merges"):
        export_model(
            model,
            tmp_path / "invalid.gguf",
            [((layout.bos_id, 105), layout.merge_base)],
            layout,
        )


@pytest.mark.parametrize("seed_count", [0, 2, 45])
@pytest.mark.integration
def test_c_uses_variable_seed_count(engine, tmp_path, seed_count):
    layout = custom_layout(seed_count)
    model = InferenceModel(ModelConfig(vocab_size=320, dim=8, heads=1, layers=1))
    path = tmp_path / "variable.gguf"
    export_model(model, path, [], layout)
    text = "seed0" if seed_count else "hello"
    expected = [layout.seed_base] if seed_count else list(b"hello")
    assert (
        list(map(int, run(engine, path, "--encode", text).stdout.split())) == expected
    )


@pytest.mark.integration
def test_c_encodes_custom_seed_and_merge_ids(engine, custom_model):
    _, path, layout, _ = custom_model
    actual = list(map(int, run(engine, path, "--encode", "seed0 hi").stdout.split()))
    assert actual == [layout.seed_base, 32, layout.merge_base]


@pytest.mark.integration
def test_c_generation_uses_custom_special_ids(engine, custom_model):
    model, path, layout, _ = custom_model
    ids = [layout.bos_id, layout.merge_base, layout.sep_id]
    pieces = {i: bytes([i]) for i in range(256)}
    pieces.update(
        {layout.seed_base + i: seed.encode() for i, seed in enumerate(layout.seeds)}
    )
    pieces[layout.merge_base] = b"hi"
    expected = bytearray()
    for _ in range(5):
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        # EOS wins ties; otherwise the smallest decodable ID wins.
        allowed = [layout.eos_id] + sorted(pieces)
        next_id = max(allowed, key=lambda i: float(logits[i]))
        if next_id == layout.eos_id:
            break
        ids.append(next_id)
        expected.extend(pieces[next_id])
    assert run(engine, path, "--generate", "hi", 5).stdout == expected + b"\n"
    with torch.no_grad():
        model.lm_head.weight.zero_()
    export_model(model, path, [((104, 105), layout.merge_base)], layout)
    result = run(engine, path, "--generate", "hi", 5)
    assert result.stdout == b"\n" and b"stop=eos" in result.stderr


@pytest.mark.parametrize(
    "field, value",
    [
        ("pad_id", 0),
        ("bos_id", 268),
        ("sep_id", 270),
        ("merge_base", 271),
        ("merge_base", 320),
        ("seed_base", 400),
    ],
)
@pytest.mark.integration
def test_c_rejects_corrupted_tokenizer_layout(
    engine, custom_model, tmp_path, field, value
):
    _, path, _, _ = custom_model
    raw = bytearray(path.read_bytes())
    key = ("capstone_sql.tokenizer." + field).encode()
    struct.pack_into("<I", raw, raw.index(key) + len(key) + 4, value)
    invalid = tmp_path / "invalid.gguf"
    invalid.write_bytes(raw)
    result = run(engine, invalid, "--encode", "hi", check=False)
    assert result.returncode != 0 and b"invalid tokenizer layout" in result.stderr


@pytest.mark.integration
def test_c_explicitly_rejects_contract_v1(engine, custom_model, tmp_path):
    _, path, _, _ = custom_model
    raw = bytearray(path.read_bytes())
    key = b"capstone_sql.contract_version"
    struct.pack_into("<I", raw, raw.index(key) + len(key) + 4, 1)
    invalid = tmp_path / "v1.gguf"
    invalid.write_bytes(raw)
    result = run(engine, invalid, "--encode", "hi", check=False)
    assert result.returncode != 0 and b"re-export as version 2" in result.stderr
