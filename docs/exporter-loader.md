# Exporter / Loader Baseline

This branch adds a small `EXL1` binary format so the Python and C sides can be
tested before the final decoder block is merged.

## Flow

1. `src/exporter.py` receives a PyTorch `state_dict` (or a small Python mapping
   in tests).
2. It writes each tensor's name, rank, shape, and little-endian `float32`
   values in insertion order.
3. `src/loader.c` validates the header and reads each record without external
   libraries. It currently prints metadata and the first value only.
4. Full token inference is intentionally deferred until the team freezes the
   decoder block, parameter names, and tensor order.

## Local check

```text
python -m pytest tests/test_exporter.py
gcc -std=c11 -O2 src/loader.c -o loader
python -c "from src.exporter import export_state_dict; export_state_dict({'w': [[1, 2], [3, 4]]}, 'dummy.bin')"
./loader dummy.bin
```

Do not commit large checkpoints or compiled binaries. The format is a baseline
contract and may be extended with model configuration after the architecture
is finalized.
