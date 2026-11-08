# 역할: 일반·gzip JSON을 읽고 폴더의 JSON 파일 목록을 모은다.
# 사용처: 데이터셋과 BPE 코퍼스를 같은 파일 규칙으로 읽는 공통 입출력에 사용한다.

"""JSON corpus access without training dependencies, including gzip archives."""
import gzip
import json
from pathlib import Path


def read_json(path: str | Path):
    path = Path(path)
    opener = gzip.open if path.name.endswith(".json.gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def json_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted([*directory.glob("*.json"), *directory.glob("*.json.gz")])
