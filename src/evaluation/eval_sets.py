"""
eval_sets.py — 평가셋 폴더와 평가 결과 위치

평가셋 하나 = 평가 폴더(config.EVAL_DIR) 아래 폴더 하나: pairs.json (질문, SQL) + sql.json (그 SQL 들의 메타데이터).
sql.json 은 손으로 만들지 않고 sql_gen 출력에서 pairs.json 에 쓰인 SQL 만 뽑아 만든다(write_sql_meta,
python -m src.run --build-sql). 폴더 하나로 완결되므로 새 평가셋은 폴더만 추가하면 된다. 모델을 쓰지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config


def list_sets(eval_dir: str | Path = config.EVAL_DIR) -> list[str]:
    """pairs.json 이 있는 하위 폴더 이름 (정렬)."""
    d = Path(eval_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if (p / config.PAIRS_FILE).is_file())


def build_sql_meta(pairs_path: str | Path, sql_path: str | Path) -> list[dict]:
    """sql_path(sql_gen 출력) 중 pairs_path 에 나오는 SQL 의 메타데이터를 원래 순서대로 반환한다."""
    with open(pairs_path, encoding="utf-8") as f:
        used = {p["sql"] for p in json.load(f)}
    with open(sql_path, encoding="utf-8") as f:
        rows = [r for r in json.load(f) if r["sql"] in used]
    missing = used - {r["sql"] for r in rows}
    if missing:
        raise ValueError(f"{sql_path} 에 없는 SQL {len(missing)}개 (예: {sorted(missing)[0]!r})")
    return rows


def write_sql_meta(name: str, sql_path: str | Path, eval_dir: str | Path = config.EVAL_DIR) -> Path:
    d = Path(eval_dir) / name
    rows = build_sql_meta(d / config.PAIRS_FILE, sql_path)
    out = d / config.EVAL_SQL_FILE
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out


def load_eval_set(name: str, eval_dir: str | Path = config.EVAL_DIR) -> tuple[list[dict], dict[str, dict]]:
    """(쌍 목록, SQL → 메타데이터). 쌍의 SQL 이 sql.json 에 하나라도 없으면 오류 — sql.json 이 낡았다는 뜻이다."""
    d = Path(eval_dir) / name
    with open(d / config.PAIRS_FILE, encoding="utf-8") as f:
        pairs = json.load(f)
    if not pairs:
        raise ValueError(f"{d / config.PAIRS_FILE} 에 평가 쌍이 없음")
    with open(d / config.EVAL_SQL_FILE, encoding="utf-8") as f:
        meta = {r["sql"]: r for r in json.load(f)}
    missing = {p["sql"] for p in pairs} - meta.keys()
    if missing:
        raise ValueError(f"{d / config.EVAL_SQL_FILE} 에 없는 SQL {len(missing)}개 — "
                         f"python -m src.run --build-sql {name} --sql <sql 파일> 로 다시 만든다")
    return pairs, meta


def default_out_dir(ckpt_path: str | Path, release_eval_dir: str | Path = config.RELEASE_EVAL_DIR) -> Path:
    """평가 결과(eval_<set>.json) 기본 위치. 출력은 항상 runs/ 아래(커밋 안 함)에 둔다:
    runs/ 안의 학습 체크포인트면 그 실행 폴더, 그 밖(배포 모델 등)이면 release_eval_dir."""
    runs_root = config.RUNS_DIR.parent.resolve()
    if runs_root in Path(ckpt_path).resolve().parents:
        return Path(ckpt_path).parent
    return Path(release_eval_dir)
