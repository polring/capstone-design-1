"""Published data and models must retain their reviewed bytes after checkout."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("group", ["stage1", "stage2", "stage3", "common"])
def test_published_bundle_hashes(group):
    manifest = json.loads((ROOT / "data/integrity.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    entries = manifest["bundles"][group]
    assert entries, f"{group}: empty integrity manifest"
    directories = [ROOT / "data" / group]
    if group != "common":
        directories.append(ROOT / "models" / group)
    actual = {path.relative_to(ROOT).as_posix()
              for directory in directories for path in directory.rglob("*") if path.is_file()}
    assert actual == set(entries), f"{group}: added={actual - set(entries)}, missing={set(entries) - actual}"
    for name, entry in entries.items():
        path = ROOT / name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], name
        if "uncompressed_sha256" in entry:
            with gzip.open(path, "rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["uncompressed_sha256"], name


def test_declared_dataset_hashes():
    """Dataset metadata must also agree with the files it describes."""
    manifests = sorted((ROOT / "data").glob("stage*/dataset_info.json"))
    assert manifests
    for manifest in manifests:
        info = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in info.values():
            if not isinstance(entry, dict) or not {"file", "sha256"} <= entry.keys():
                continue
            path = manifest.parent / entry["file"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], str(path)
            if "uncompressed_sha256" in entry:
                with gzip.open(path, "rb") as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["uncompressed_sha256"], str(path)
