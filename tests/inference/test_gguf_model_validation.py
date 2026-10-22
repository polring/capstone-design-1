"""Boundary tests for the named exporter/inference validation predicates."""

import pytest
import torch

from src.exporter import _is_valid_ordered_merge
from src.inference.gguf_model import (
    ModelConfig,
    _has_checkpoint_contract,
    _has_training_checkpoint_format,
    _is_input_shape_within_context,
    from_training_checkpoint,
)
from src.models.model import TextToSQLModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"heads": 0}, "dimensions must be positive integers"),
        ({"dim": True}, "dimensions must be positive integers"),
        ({"layers": -1}, "dimensions must be positive integers"),
        ({"context": 1.5}, "dimensions must be positive integers"),
        ({"vocab_size": 255}, "vocab >= 256"),
        ({"dim": 15, "heads": 2}, "vocab >= 256"),
        ({"dim": 12, "heads": 4}, "vocab >= 256"),
        ({"eps": 0}, "eps and theta must be finite and positive"),
        ({"theta": float("inf")}, "eps and theta must be finite and positive"),
        ({"eps": float("nan")}, "eps and theta must be finite and positive"),
    ],
)
def test_config_validation_preserves_rejection_messages(changes, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**changes)


def test_config_accepts_minimum_vocabulary_and_even_head_dimension():
    config = ModelConfig(vocab_size=256, dim=8, heads=4)
    assert config._has_positive_integer_dimensions()
    assert config._has_compatible_attention_dimensions()
    assert config._has_valid_numeric_parameters()


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((1, 1), True),
        ((1, 4), True),
        ((1, 5), False),
        ((1, 0), False),
        ((4,), False),
        ((), False),
        ((1, 2, 3), False),
    ],
)
def test_input_shape_checks_rank_before_sequence_length(shape, expected):
    assert _is_input_shape_within_context(torch.empty(shape), 4) is expected


@pytest.mark.parametrize(
    "checkpoint, expected",
    [
        (None, False),
        ([], False),
        ({}, False),
        ({"config": {}}, False),
        ({"state_dict": {}}, False),
        ({"config": {}, "state_dict": {}}, True),
        ({"config": {}, "state_dict": {}, "epoch": 1}, True),
    ],
)
def test_checkpoint_contract_requires_both_keys(checkpoint, expected):
    assert _has_checkpoint_contract(checkpoint) is expected


@pytest.mark.parametrize(
    "checkpoint, expected",
    [
        (None, False),
        ({"model": {}}, False),
        ({"model_cfg": {}}, False),
        ({"config": {}, "state_dict": {}}, False),
        ({"model_cfg": {}, "model": {}}, True),
        ({"format_version": 1, "stage": 1, "model_cfg": {}, "model": {}}, True),
    ],
)
def test_training_checkpoint_format_requires_both_keys(checkpoint, expected):
    assert _has_training_checkpoint_format(checkpoint) is expected


@pytest.mark.parametrize("valid_vocab_size", [None, 300, 320])
def test_training_checkpoint_converts_to_same_logits(valid_vocab_size):
    torch.manual_seed(0)
    cfg = {
        "vocab_size": 320,
        "dim": 16,
        "num_layers": 2,
        "num_heads": 2,
        "ffn_dim": 43,
        "max_seq_len": 32,
        "padding_idx": 256,
        "valid_vocab_size": valid_vocab_size,
    }
    team = TextToSQLModel(**cfg).eval()
    converted = from_training_checkpoint({"model_cfg": cfg, "model": team.state_dict()})
    assert converted.config.layers == 2 and converted.config.context == 32
    ids = torch.tensor([[257, 104, 105, 259, 299]])
    with torch.no_grad():
        torch.testing.assert_close(converted(ids), team(ids), rtol=0, atol=0)
    assert converted.config.vocab_size == (valid_vocab_size or cfg["vocab_size"])


def test_unused_output_token_cannot_win_after_conversion():
    cfg = dict(vocab_size=320, valid_vocab_size=300, dim=16, num_layers=1,
               num_heads=2, ffn_dim=24, max_seq_len=8)
    team = TextToSQLModel(**cfg).eval()
    ids = torch.tensor([[257, 104, 259]])
    hidden = []
    handle = team.norm.register_forward_hook(lambda module, args, output: hidden.append(output))
    with torch.no_grad():
        team(ids)
        handle.remove()
        team.lm_head.weight[319].copy_(hidden[0][0, -1] * 1000)
        # This unused row would beat every valid token without the vocabulary restriction.
        assert team.lm_head(hidden[0])[0, -1].argmax().item() == 319
        converted = from_training_checkpoint({"model_cfg": cfg, "model": team.state_dict()})
        torch.testing.assert_close(converted(ids), team(ids))
        assert converted(ids).shape[-1] == 300


@pytest.mark.parametrize("invalid", [0, -1, 321, 300.5, True])
def test_training_conversion_rejects_invalid_vocabulary(invalid):
    with pytest.raises(ValueError, match="valid_vocab_size"):
        from_training_checkpoint({"model_cfg": {"vocab_size": 320, "valid_vocab_size": invalid},
                                  "model": {}})


@pytest.mark.parametrize(
    "merge, index, vocab_size, expected",
    [
        (((104, 105), 298), 0, 300, True),
        (((298, 105), 299), 1, 300, True),
        (((999, 105), 298), 0, 300, False),
        (((104, 999), 298), 0, 300, False),
        (((104, 105), 299), 0, 300, False),
        (((104, 105), 298), 0, 298, False),
    ],
)
def test_ordered_merge_checks_references_order_and_vocabulary(
    merge, index, vocab_size, expected
):
    known = set(range(256)) | set(range(260, 298))
    if index == 1:
        known.add(298)
    assert _is_valid_ordered_merge(merge, index, known, vocab_size, 298) is expected
