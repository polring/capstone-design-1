# 역할: 검토한 데이터·모델 파일의 SHA-256과 압축 해제 내용의 해시를 기록한다.
# 사용처: python -m src.data.integrity로 data/integrity.json을 생성할 때 사용한다.

"""Record reviewed data/model bytes: python -m src.data.integrity."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/integrity.json"


def bundle_files(group: str) -> list[Path]:
    directories = [ROOT / "data" / group]
    if group != "common":
        directories.append(ROOT / "models" / group)
    return sorted(path for directory in directories for path in directory.rglob("*")
                  if path.is_file())


def main() -> None:
    groups = sorted(path.name for path in (ROOT / "data").glob("stage[0-9]*")
                    if path.is_dir())
    if (ROOT / "data/common").is_dir():
        groups.append("common")
    bundles = {}
    for group in groups:
        entries = {}
        for path in bundle_files(group):
            with path.open("rb") as stream:
                entry = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
            if path.suffix == ".gz":
                with gzip.open(path, "rb") as stream:
                    entry["uncompressed_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            entries[path.relative_to(ROOT).as_posix()] = entry
        bundles[group] = entries
    MANIFEST.write_text(json.dumps({"format_version": 1, "bundles": bundles},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{MANIFEST.relative_to(ROOT)}: {sum(map(len, bundles.values()))} files")


if __name__ == "__main__":
    main()
