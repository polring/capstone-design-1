"""Boundary tests for the named exporter/inference validation predicates."""

import pytest
import torch

from src.exporter import _is_valid_ordered_merge
from src.inference import (
    ModelConfig,
    _has_checkpoint_contract,
    _is_input_shape_within_context,
)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"heads": 0}, "dimensions must be positive integers"),
        ({"dim": True}, "dimensions must be positive integers"),
        ({"layers": -1}, "dimensions must be positive integers"),
        ({"context": 1.5}, "dimensions must be positive integers"),
        ({"vocab_size": 297}, "vocab >= 298"),
        ({"dim": 15, "heads": 2}, "vocab >= 298"),
        ({"dim": 12, "heads": 4}, "vocab >= 298"),
        ({"eps": 0}, "eps and theta must be finite and positive"),
        ({"theta": float("inf")}, "eps and theta must be finite and positive"),
        ({"eps": float("nan")}, "eps and theta must be finite and positive"),
    ],
)
def test_config_validation_preserves_rejection_messages(changes, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**changes)


def test_config_accepts_minimum_vocabulary_and_even_head_dimension():
    config = ModelConfig(vocab_size=298, dim=8, heads=4)
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
    assert _is_valid_ordered_merge(merge, index, known, vocab_size) is expected
