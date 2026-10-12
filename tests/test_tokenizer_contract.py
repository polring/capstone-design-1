"""Contract v2 must not depend on the stage-1 tokenizer's ID layout."""

from dataclasses import asdict, replace
import json
import struct

import pytest
import torch

from src.exporter import export_model, read_gguf
from src.inference.gguf_model import InferenceModel, ModelConfig
from src.tokenizer.contract import TokenizerConfig


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
@pytest.mark.unit
def test_export_rejects_invalid_layout_before_writing(custom_model, tmp_path, changes):
    model, _, layout, _ = custom_model
    path = tmp_path / "invalid.gguf"
    with pytest.raises(ValueError):
        export_model(model, path, [], replace(layout, **changes))
    assert not path.exists()


@pytest.mark.unit
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
def test_read_gguf_follows_stored_layout(tmp_path, seed_count):
    """파일에 저장된 토큰 ID 배치(seed 개수, 시작 ID, 특수 토큰)대로 다시 인코딩한다.

    C 엔진 쪽 같은 검사는 C 엔진 저장소의 fixture(custom_layout, no_seeds, many_seeds)와 오류 테스트가 맡는다.
    """
    layout = custom_layout(seed_count)
    merges = [((104, 105), layout.merge_base)]
    model = InferenceModel(ModelConfig(vocab_size=320, dim=8, heads=1, layers=1))
    path = tmp_path / "layout.gguf"
    export_model(model, path, merges, layout)
    _, tokenizer, _ = read_gguf(path)
    assert tokenizer.config == layout
    assert (tokenizer.bos_id, tokenizer.eos_id, tokenizer.sep_id) == (267, 268, 269)
    expected = [layout.seed_base, 32] if seed_count else list(b"seed0 ")
    assert tokenizer.encode("seed0 hi") == expected + [layout.merge_base]
