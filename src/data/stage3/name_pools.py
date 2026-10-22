# 역할: 두 단어 고객·상품 이름 풀과 DB 이름과 겹치지 않는 교체 후보를 구성한다.
# 사용처: 3단계 DB 이름 준비와 학습 중 범주를 유지하는 이름 교체에 사용한다.

"""Stage-3 two-word source pools and DB-disjoint replacement candidates.

python -m src.data.stage3.name_pools --db data/stage3/db/shop.db
"""

import argparse
import json
import sqlite3
from pathlib import Path
from src.config import stage_paths

PATHS = stage_paths(3)
POOL_PATH = PATHS.data_dir / "name_pools.json"


def load_pools(path=POOL_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = [data["given_names"], data["surnames"]]
    if set(data["category_nouns"]) != set(data["category_modifiers"]):
        raise ValueError("Product categories differ between nouns and modifiers")
    groups += list(data["category_nouns"].values())
    groups += list(data["category_modifiers"].values())
    for words in groups:
        if not words or len(words) != len(set(words)):
            raise ValueError("Pool must be nonempty and unique")
        if any(not w.isascii() or not w.isalpha() or not w.islower() for w in words):
            raise ValueError("Pool components must be lowercase ASCII words")
    return data


def customer_names(pools):
    return [f"{first} {last}" for first in pools["given_names"] for last in pools["surnames"]]


def product_names(pools):
    return {
        category: [f"{modifier} {noun}" for modifier in pools["category_modifiers"][category]
                   for noun in nouns]
        for category, nouns in pools["category_nouns"].items()
    }


def replacement_pools(db, pools):
    with sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True) as con:
        names = {r[0] for r in con.execute("SELECT name FROM customers")}
        items = {r[0] for r in con.execute("SELECT item_name FROM items")}
    return {
        "name": [n for n in customer_names(pools) if n not in names],
        "item_name_by_category": {
            c: [n for n in values if n not in items] for c, values in product_names(pools).items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(PATHS.db_path))
    parser.add_argument("--pools", default=str(POOL_PATH))
    parser.add_argument("--out", default=str(PATHS.raw_dir / "name_pools/replacement_pools.json"))
    args = parser.parse_args(argv)
    candidates = replacement_pools(args.db, load_pools(args.pools))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: {len(candidates['name'])} customer replacements; "
          f"{sum(map(len, candidates['item_name_by_category'].values()))} product replacements")


if __name__ == "__main__":
    main()
