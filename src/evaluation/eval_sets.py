# 역할: 평가 질문·정답 SQL·메타데이터를 읽고 평가셋과 결과 경로를 관리한다.
# 사용처: 단계별 평가셋 조회·생성과 학습·추론의 평가 입력 로딩에 사용한다.

"""
eval_sets.py — 평가셋 파일과 평가 결과 위치

평가셋은 평가 폴더(config.EVAL_DIR)에 두 가지 방식으로 둔다. 둘 다 pairs.json (질문, SQL) + sql.json (그 SQL 들의 메타데이터)이다.
- 폴더 방식(1단계): 평가셋 하나 = 하위 폴더 하나(<eval_dir>/<이름>/pairs.json + sql.json). 새 평가셋은 폴더만 추가하면 된다.
- 합친 방식(2단계): <eval_dir>/pairs.json 하나에 모든 평가셋의 쌍을 담고 쌍마다 set(평가셋 이름)을 적는다. sql.json 도 하나다.
  eval_build 가 write_combined 로 만든다.
불러올 때는 두 방식 모두 평가셋 이름으로 나뉘어(list_sets, load_eval_set) 평가셋마다 따로 채점한다.
sql.json 은 손으로 만들지 않고 sql_gen 출력에서 쌍에 쓰인 SQL 만 뽑아 만든다(write_sql_meta / write_combined,
python -m src.run --build-sql). 모델을 쓰지 않는다.
쌍에 sql_group(정규형 SQL)이 있으면 메타데이터는 그것으로 찾는다: 2단계 쌍의 sql 은 IN 값을 질문에 나온 순서로 써서
sql_gen 출력(정규형)과 문자열이 다를 수 있다.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src import config

SET_KEY = "set"  # 합친 방식에서 쌍이 속한 평가셋 이름


@lru_cache(maxsize=8)
def _read_json(path: str, mtime: float) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read(path: Path) -> list:
    """같은 파일을 평가셋마다 다시 읽지 않는다 (합친 pairs.json 은 수 MB). 파일이 바뀌면 다시 읽는다."""
    return _read_json(str(path), path.stat().st_mtime)


def _combined_pairs(eval_dir: str | Path) -> list[dict]:
    f = Path(eval_dir) / config.PAIRS_FILE
    return _read(f) if f.is_file() else []


def list_sets(eval_dir: str | Path = config.EVAL_DIR) -> list[str]:
    """평가셋 이름 (정렬): pairs.json 이 있는 하위 폴더 + 합친 pairs.json 의 set 이름."""
    d = Path(eval_dir)
    if not d.is_dir():
        return []
    names = {p.name for p in d.iterdir() if (p / config.PAIRS_FILE).is_file()}
    return sorted(names | {p[SET_KEY] for p in _combined_pairs(d)})


def sql_key(pair: dict) -> str:
    """쌍의 SQL 메타데이터 키: sql_group(정규형 SQL)이 있으면 그것, 없으면 sql."""
    return pair.get("sql_group", pair["sql"])


def build_sql_meta(pairs: str | Path | list[dict], sql_paths: str | Path | list) -> list[dict]:
    """sql_paths(sql_gen 출력 파일 하나 또는 여러 개) 중 pairs(쌍 목록 또는 pairs.json 경로)에 나오는 SQL 의 메타데이터를
    원래 순서대로 반환한다. 같은 SQL 이 여러 파일에 있으면 앞 파일 것을 쓴다."""
    if isinstance(sql_paths, (str, Path)):
        sql_paths = [sql_paths]
    if isinstance(pairs, (str, Path)):
        with open(pairs, encoding="utf-8") as f:
            pairs = json.load(f)
    used = {sql_key(p) for p in pairs}
    rows, seen = [], set()
    for path in sql_paths:
        with open(path, encoding="utf-8") as f:
            for r in json.load(f):
                if r["sql"] in used and r["sql"] not in seen:
                    rows.append(r)
                    seen.add(r["sql"])
    missing = used - seen
    if missing:
        raise ValueError(f"{', '.join(map(str, sql_paths))} 에 없는 SQL {len(missing)}개 (예: {sorted(missing)[0]!r})")
    return rows


def write_sql_meta(name: str, sql_paths: str | Path | list, eval_dir: str | Path = config.EVAL_DIR) -> Path:
    d = Path(eval_dir) / name
    rows = build_sql_meta(d / config.PAIRS_FILE, sql_paths)
    out = d / config.EVAL_SQL_FILE
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return out


def write_combined(sets: dict[str, list[dict]], sql_paths: str | Path | list, eval_dir: str | Path,
                   drop: set[str] = frozenset()) -> Path:
    """합친 방식: <eval_dir>/pairs.json 에서 sets 의 평가셋을 새 쌍으로 바꾸고(없으면 추가) drop 의 평가셋은 지운 뒤,
    sql.json 을 남은 쌍 전부로 다시 만든다. sql_paths 는 모든 평가셋의 SQL 을 담아야 한다. 쌍은 평가셋 이름 순서로 쓴다."""
    d = Path(eval_dir)
    d.mkdir(parents=True, exist_ok=True)
    by_set: dict[str, list[dict]] = {}
    for p in _combined_pairs(d):
        by_set.setdefault(p[SET_KEY], []).append(p)
    for name in drop:
        by_set.pop(name, None)
    by_set.update({name: [dict(p, **{SET_KEY: name}) for p in pairs] for name, pairs in sets.items()})
    pairs = [p for name in sorted(by_set) for p in by_set[name]]
    rows = build_sql_meta(pairs, sql_paths)
    with open(d / config.PAIRS_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    with open(d / config.EVAL_SQL_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return d / config.PAIRS_FILE


def load_eval_set(name: str, eval_dir: str | Path = config.EVAL_DIR) -> tuple[list[dict], dict[str, dict]]:
    """(쌍 목록, SQL → 메타데이터: 그 평가셋에 쓰인 SQL 만). 폴더 방식이 먼저, 없으면 합친 pairs.json 의 set 으로 고른다.
    쌍의 SQL 이 sql.json 에 하나라도 없으면 오류 — sql.json 이 낡았다는 뜻이다."""
    d = Path(eval_dir) / name
    if (d / config.PAIRS_FILE).is_file():
        pairs, sql_file = [dict(p) for p in _read(d / config.PAIRS_FILE)], d / config.EVAL_SQL_FILE
        rebuild = f"python -m src.run --build-sql {name} --sql <sql 파일>"
    else:
        pairs = [{k: v for k, v in p.items() if k != SET_KEY} for p in _combined_pairs(eval_dir) if p[SET_KEY] == name]
        sql_file, rebuild = Path(eval_dir) / config.EVAL_SQL_FILE, "평가셋을 만든 명령(2단계: eval_build)"
        if not pairs:
            raise FileNotFoundError(f"{eval_dir} 에 평가셋 {name} 이 없음 (폴더도, 합친 pairs.json 의 set 도 없음)")
    if not pairs:
        raise ValueError(f"{eval_dir} 에 평가셋 {name} 의 쌍이 없음")
    used = {sql_key(p) for p in pairs}
    meta = {r["sql"]: dict(r) for r in _read(sql_file) if r["sql"] in used}  # 읽은 파일은 캐시에 남으므로 사본을 준다
    missing = used - meta.keys()
    if missing:
        raise ValueError(f"{sql_file} 에 없는 SQL {len(missing)}개 — {rebuild}로 다시 만든다")
    return pairs, meta


def default_out_dir(ckpt_path: str | Path, release_eval_dir: str | Path = config.RELEASE_EVAL_DIR) -> Path:
    """평가 결과(eval_<set>.json) 기본 위치. 출력은 항상 runs/ 아래(커밋 안 함)에 둔다:
    runs/ 안의 학습 체크포인트면 그 실행 폴더, 그 밖(배포 모델 등)이면 release_eval_dir."""
    runs_root = config.RUNS_DIR.parent.resolve()
    if runs_root in Path(ckpt_path).resolve().parents:
        return Path(ckpt_path).parent
    return Path(release_eval_dir)
