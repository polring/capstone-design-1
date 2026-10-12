"""GGUF v3 writer, little endian F32 with 32-byte alignment.

capstone_sql is our custom architecture, not an automatically llama.cpp-supported
model. The writer uses the standard library, PyTorch and the team's tokenizer.
"""

import argparse
import io
import struct
from pathlib import Path
import torch
from src.inference.gguf_model import (
    InferenceModel,
    ModelConfig,
    _has_training_checkpoint_format,
    load_checkpoint,
)
from src.inference.loading import tokenizer_path_for
from src.tokenizer import bpe
from src.tokenizer.contract import CONTRACT_VERSION, TokenizerConfig


def _string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _is_valid_ordered_merge(merge, index, known, vocab_size, merge_base):
    """A merge must use existing tokens and the next available vocabulary ID."""
    (a, b), new = merge
    return a in known and b in known and new == merge_base + index and new < vocab_size


def export_model(model, output, merges, tokenizer_config=None):
    config = model.config
    tokenizer_config = tokenizer_config or TokenizerConfig.from_team_bpe()
    tokenizer_config.validate(config.vocab_size)
    known = tokenizer_config.known_token_ids()
    flat = []
    for index, merge in enumerate(merges):
        if not _is_valid_ordered_merge(
            merge, index, known, config.vocab_size, tokenizer_config.merge_base
        ):
            raise ValueError("invalid ordered tokenizer merges")
        (a, b), new = merge
        known.add(new)
        flat.extend((a, b, new))
    metadata = [
        ("general.architecture", 8, _string("capstone_sql")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("capstone_sql.contract_version", 4, struct.pack("<I", CONTRACT_VERSION)),
    ]
    for name in ("vocab_size", "dim", "layers", "heads", "ffn_dim", "context"):
        metadata.append(
            ("capstone_sql." + name, 4, struct.pack("<I", getattr(config, name)))
        )
    for name in ("eps", "theta"):
        metadata.append(
            ("capstone_sql." + name, 6, struct.pack("<f", getattr(config, name)))
        )
    metadata += [
        (
            "capstone_sql.tokenizer.seeds",
            9,
            struct.pack("<IQ", 8, len(tokenizer_config.seeds))
            + b"".join(map(_string, tokenizer_config.seeds)),
        ),
        (
            "capstone_sql.tokenizer.merges",
            9,
            struct.pack("<IQ", 4, len(flat)) + struct.pack(f"<{len(flat)}I", *flat),
        ),
    ]
    for name in ("pad_id", "bos_id", "eos_id", "sep_id", "seed_base", "merge_base"):
        metadata.append(
            (
                "capstone_sql.tokenizer." + name,
                4,
                struct.pack("<I", getattr(tokenizer_config, name)),
            )
        )
    expected = InferenceModel(config).state_dict()
    state = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("state_dict names do not match inference contract")
    header = io.BytesIO()
    header.write(struct.pack("<4sIQQ", b"GGUF", 3, len(state), len(metadata)))
    for key, kind, value in metadata:
        header.write(_string(key) + struct.pack("<I", kind) + value)
    tensor_data = bytearray()
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"wrong shape: {name}")
        tensor = tensor.detach().cpu().to(torch.float32).contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite tensor: {name}")
        # GGUF dimensions are fastest-first; bytes remain PyTorch row-major.
        dims = tuple(reversed(tensor.shape))
        header.write(_string(name) + struct.pack("<I", len(dims)))
        header.write(struct.pack(f"<{len(dims)}Q", *dims))
        header.write(struct.pack("<IQ", 0, len(tensor_data)))
        tensor_data.extend(tensor.numpy().astype("<f4", copy=False).tobytes())
        tensor_data.extend(b"\0" * (-len(tensor_data) % 32))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(header.getvalue())
        stream.write(b"\0" * (-header.tell() % 32))
        stream.write(tensor_data)


def read_gguf(path):
    """Read a file written by export_model back into (model, tokenizer, metadata).

    Only the subset of GGUF that export_model writes is supported; the contract
    fixtures use it to recompute expected values from the committed model file.
    """
    raw = Path(path).read_bytes()
    position = 0

    def take(fmt):
        nonlocal position
        values = struct.unpack_from(fmt, raw, position)
        position += struct.calcsize(fmt)
        return values if len(values) > 1 else values[0]

    def take_string():
        nonlocal position
        length = take("<Q")
        position += length
        return raw[position - length : position].decode("utf-8")

    def take_value(kind):
        if kind == 4:
            return take("<I")
        if kind == 6:
            return take("<f")
        if kind == 8:
            return take_string()
        element_kind, count = take("<IQ")
        return [take_value(element_kind) for _ in range(count)]

    magic, version, tensor_count, metadata_count = take("<4sIQQ")
    if magic != b"GGUF" or version != 3:
        raise ValueError("not a GGUF v3 file")
    metadata = {}
    for _ in range(metadata_count):
        key = take_string()
        metadata[key] = take_value(take("<I"))
    descriptors = []
    for _ in range(tensor_count):
        name = take_string()
        dims = take(f"<{take('<I')}Q")
        dims = dims if isinstance(dims, tuple) else (dims,)
        _, offset = take("<IQ")
        descriptors.append((name, tuple(reversed(dims)), offset))
    data_start = position + (-position % 32)

    def field(name):
        return metadata["capstone_sql." + name]

    config = ModelConfig(
        **{
            name: field(name)
            for name in ("vocab_size", "dim", "layers", "heads", "ffn_dim", "context")
        },
        eps=field("eps"),
        theta=field("theta"),
    )
    state = {}
    for name, shape, offset in descriptors:
        count = 1
        for size in shape:
            count *= size
        start = data_start + offset
        values = struct.unpack_from(f"<{count}f", raw, start)
        state[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
    model = InferenceModel(config)
    model.load_state_dict(state, strict=True)
    flat = field("tokenizer.merges")
    merges = [
        ((flat[index], flat[index + 1]), flat[index + 2])
        for index in range(0, len(flat), 3)
    ]
    tokenizer = bpe.Tokenizer(field("tokenizer.seeds"), merges)
    return model.eval(), tokenizer, metadata


def _checkpoint_tokenizer_path(checkpoint_path):
    """The tokenizer a training or release checkpoint was trained with, or None."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not _has_training_checkpoint_format(checkpoint):
        return None
    return tokenizer_path_for(checkpoint_path, checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export capstone_sql F32 GGUF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--dummy", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer",
        help="team tokenizer.json (seeds + merges); defaults to the one saved"
        " with a training or release checkpoint",
    )
    parser.add_argument(
        "--tokenizer-config", help="JSON token layout matching the training tokenizer"
    )
    args = parser.parse_args(argv)
    if args.dummy:
        torch.manual_seed(17)
        model = InferenceModel(
            ModelConfig(dim=16, heads=2, layers=2, ffn_dim=32)
        ).eval()
        tokenizer_path = args.tokenizer
    else:
        tokenizer_path = args.tokenizer or _checkpoint_tokenizer_path(args.checkpoint)
        if not tokenizer_path:
            parser.error(
                "--checkpoint in the config/state_dict format requires --tokenizer"
            )
        model = load_checkpoint(args.checkpoint)
    # Tokenizer.load also reads the older merges-only bpe_merges.json.
    tokenizer = (
        bpe.Tokenizer.load(tokenizer_path)
        if tokenizer_path
        else bpe.Tokenizer(bpe.load_seed_tokens())
    )
    tokenizer_config = (
        TokenizerConfig.load(args.tokenizer_config)
        if args.tokenizer_config
        else TokenizerConfig.from_tokenizer(tokenizer)
    )
    export_model(model, args.output, tokenizer.merges, tokenizer_config)
    print(f"saved {args.output}; architecture=capstone_sql; dummy={args.dummy}")


if __name__ == "__main__":
    main()
