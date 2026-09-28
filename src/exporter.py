"""Export model tensors to the small binary format consumed by the C loader.

The format is deliberately simple for the baseline:

    magic (4 bytes: ``EXL1``)
    tensor_count (uint32, little endian)
    repeated tensor records:
      name_length (uint32), name (UTF-8)
      ndim (uint32), shape (ndim * uint32)
      values (shape product * float32, little endian)

Keeping the header and tensor order explicit makes the eventual PyTorch model
and the standalone C inference engine independently testable.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Mapping, Sequence

MAGIC = b"EXL1"
VERSION = 1
_U32 = struct.Struct("<I")


def _as_flat_float32(values: object) -> tuple[tuple[int, ...], bytes]:
    """Return a tensor-like object's shape and little-endian float32 bytes."""
    shape = tuple(int(x) for x in getattr(values, "shape", ()))
    if hasattr(values, "detach"):
        values = values.detach().cpu().contiguous().numpy()
        shape = tuple(int(x) for x in values.shape)
        raw = values.astype("<f4", copy=False).tobytes()
        return shape, raw

    # Standard-library fallback used by the dummy test and useful for simple
    # lists before torch is available in the project environment.
    if not shape:
        values = list(values)  # type: ignore[arg-type]

        def infer_shape(item: object) -> tuple[int, ...]:
            if not isinstance(item, (list, tuple)):
                return ()
            return (len(item),) + (infer_shape(item[0]) if item else ())

        shape = infer_shape(values)
    flat: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            flat.append(float(item))

    visit(values)
    return shape, struct.pack(f"<{len(flat)}f", *flat)


def export_state_dict(state_dict: Mapping[str, object], output: str | Path) -> None:
    """Write a mapping of tensor names to tensor-like values as ``output``."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(_U32.pack(VERSION))
        stream.write(_U32.pack(len(state_dict)))
        for name, values in state_dict.items():
            encoded_name = name.encode("utf-8")
            shape, raw = _as_flat_float32(values)
            stream.write(_U32.pack(len(encoded_name)))
            stream.write(encoded_name)
            stream.write(_U32.pack(len(shape)))
            for dimension in shape:
                stream.write(_U32.pack(dimension))
            stream.write(raw)


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> None:
    """Load a PyTorch checkpoint and export its ``state_dict``.

    PyTorch remains an optional dependency for this baseline; importing it only
    happens when this checkpoint-oriented entry point is called.
    """
    import torch

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    export_state_dict(state_dict, output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a PyTorch checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
