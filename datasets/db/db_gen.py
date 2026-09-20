"""
Text-to-SQL 1단계 DB 생성기 (v2)

v1 대비 변경점
  1. 이름을 한 단어로 (customers.name, items.item_name)
  2. 행 수 확장: 300 / 200 / 500  (preset "large", 기본값)
  3. 카테고리별 이름 풀 고갈 방지 (balanced_column 에 cap 추가)
  4. `space` 명령 추가: 1단계 문법으로 만들 수 있는 고유 SQL 수를 계산

사용법
    python db_gen.py build                      # preset=large, seed=0, out=./data
    python db_gen.py build --preset small --seed 7
    python db_gen.py verify
    python db_gen.py test                       # 분포/결정성/price등호/미등장값 4종
    python db_gen.py summary
    python db_gen.py space                      # SQL 공간과 ID 조건 비중

산출물
    data/shop.db        SQLite DB
    data/holdout.json   미등장 값 목록 (SQL 생성기가 읽어 해당 리터럴을 제외)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# 값 풀
# ---------------------------------------------------------------------------

CITIES = ["new york", "chicago", "houston", "seattle", "boston", "denver"]
MEMBERSHIPS = ["bronze", "silver", "gold", "platinum"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]

# 고객 이름: 한 단어. 300개.
GIVEN_NAMES = """
aaron abigail adam adrian aiden alan albert alexis alice allison
amanda amber amy andrea andrew angela anna anthony april arthur
ashley audrey austin autumn barbara beatrice benjamin bernard bethany betty
rachel raymond rebecca regina renee rhonda richard rita robert roberta
roger ronald rosalind rose ross roy ruby russell ruth ryan
samuel sandra sara savannah scarlett scott sebastian sergio seth shannon
sharon shirley sidney silas simon sophia spencer stacy stanley stella
stephen steven stuart susan sylvia tamara tanya taylor terrence theodore
thomas tiffany timothy tobias tracy travis trevor tristan tyler ursula
valerie vanessa vera victor victoria vincent viola virginia vivian walter
blake bonnie brandon brenda brian brittany bruce bryan caleb cameron
carl carla carmen carol carrie casey catherine cecilia chad charles
charlotte chelsea cheryl chloe christina christopher clara clayton clifford colin
connor corey courtney craig crystal curtis cynthia daisy dale dana
daniel danielle darren david dawn dean deborah declan denise dennis
derek desmond diana diane dominic donald donna dorothy douglas dustin
dylan edgar edith edward eileen elaine eleanor elena eliza elizabeth
ellen emily emma eric erica erin ernest ethan eugene eva
evelyn faith felix fernando fiona florence frances francis frank franklin
fred gabriel gail garrett gary gavin gemma george georgia gerald
gilbert gina glenn gloria gordon grace graham grant gregory gwen
hannah harold harriet harvey hayden heather hector helen henry herbert
hilary holly howard hugh hunter ian imogen irene iris isaac
isabel ivan jacob jade james jamie jane janet jared jasmine
jason jasper jean jeffrey jenna jennifer jeremy jerome jesse jessica
joan joanna joel john jonathan jordan joseph joshua joyce judith
julia julian june justin karen karl kate katherine kathleen kayla
keith kelly kenneth kevin kimberly kirsten kyle lance larry laura
lauren lawrence leah lena leo leonard leslie lewis liam lillian
linda lindsay lionel lisa logan lois lorraine louis lucas lucy
luke lydia lynn madeline madison malcolm marcus margaret maria marilyn
mario marion mark marlene martha martin marvin mary mason matthew
maureen maurice maxwell megan melanie melissa meredith micah michael michelle
miles miranda miriam mitchell molly monica morgan nadia nancy naomi
natalie nathan neil nelson nicholas nicole nigel noah nora norman
olivia oliver oscar owen paige pamela patricia patrick paul paula
pearl pedro peggy penelope percy perry peter philip phoebe phyllis
""".split()

# 상품명: 한 단어. 카테고리별 22개.
CATEGORY_NOUNS: dict[str, list[str]] = {
    "electronics": """mouse keyboard monitor headset webcam speaker charger router laptop tablet
                      printer scanner projector microphone adapter modem drone camera console
                      earbuds smartwatch powerbank
                      dongle hub dock cable battery remote joystick gamepad receiver transmitter
                      amplifier equalizer turntable radio television flashdrive harddrive chipset""".split(),
    "office supplies": """notebook stapler binder marker whiteboard envelope folder calculator
                          clipboard highlighter eraser ruler scissors tape calendar planner pencil
                          pen sharpener organizer shredder laminator
                          stamp inkpad paperclip thumbtack bookend binderclip letterhead notepad
                          postit rubberband glue chalk pushpin divider portfolio journal ledger punch""".split(),
    "kitchen": """kettle blender toaster mug plate bowl spatula whisk grater strainer ladle
                  skillet saucepan teapot thermos corkscrew peeler colander tumbler pitcher
                  tray steamer
                  fridge oven microwave dishwasher sieve funnel rollingpin cuttingboard knife fork
                  spoon chopstick jar canister decanter carafe trivet apron""".split(),
    "furniture": """chair desk bookshelf stool cabinet wardrobe bench sofa dresser nightstand
                    ottoman recliner futon hammock mattress headboard credenza armchair sideboard
                    shelf table locker
                    couch loveseat barstool daybed bunkbed crib vanity bureau hutch cupboard
                    drawer partition screen rack stand pedestal footstool chaise""".split(),
    "clothing": """jacket hoodie scarf cap gloves socks raincoat sweater cardigan blazer jeans
                   shorts skirt dress vest poncho beanie mittens sandals boots sneakers belt
                   coat parka tunic robe pajamas legging tights jumpsuit romper kimono turtleneck
                   poloshirt tshirt tanktop camisole overalls cape shawl""".split(),
    "sports": """dumbbell helmet racket kettlebell treadmill skateboard bicycle snorkel paddle
                 javelin frisbee goggles barbell trampoline surfboard kayak canoe sled skis
                 snowboard bat mitt
                 jumprope yogamat medicineball punchingbag stopwatch whistle cleats pedometer
                 canteen harness carabiner compass binoculars lifejacket wetsuit snowshoe discus shotput""".split(),
}
CATEGORIES = list(CATEGORY_NOUNS.keys())

# 두 단어 모드(name_style="two_word")에서만 사용
CATEGORY_MODIFIERS: dict[str, list[str]] = {
    "electronics": ["wireless", "bluetooth", "portable", "compact", "gaming", "usb"],
    "office supplies": ["leather", "plastic", "metal", "magnetic", "adjustable", "slim"],
    "kitchen": ["stainless", "ceramic", "electric", "nonstick", "glass", "copper"],
    "furniture": ["wooden", "folding", "oak", "padded", "corner", "standing"],
    "clothing": ["cotton", "wool", "denim", "fleece", "linen", "waterproof"],
    "sports": ["rubber", "foam", "insulated", "lightweight", "carbon", "mesh"],
}

# 풀에 중복이 섞이면 uniqueness 제약에서 무한루프가 난다. 즉시 잡는다.
assert len(GIVEN_NAMES) == len(set(GIVEN_NAMES)), "GIVEN_NAMES 중복"
_nouns = [n for ns in CATEGORY_NOUNS.values() for n in ns]
assert len(_nouns) == len(set(_nouns)), "CATEGORY_NOUNS 중복"


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_customers: int = 300
    n_items: int = 200
    n_orders: int = 500

    # 테이블별 분리 ID 대역 (5-2)
    customer_id_range: tuple[int, int] = (1000, 1999)
    item_id_range: tuple[int, int] = (3000, 3999)
    order_id_range: tuple[int, int] = (5000, 5999)

    # "single" = 한 단어(james, mouse) / "two_word" = 두 단어(wireless mouse)
    name_style: str = "single"

    min_per_city: int = 38
    min_per_membership: int = 56
    min_per_category: int = 25
    min_per_quantity: int = 27
    min_per_status: int = 83
    min_zero_stock: int = 13

    price_min: float = 1.00
    price_max: float = 199.99
    stock_min: int = 0
    stock_max: int = 100
    quantity_values: tuple[int, ...] = tuple(range(1, 11))

    # 미등장 값 분리 비율 (5-2)
    heldout_customer_ratio: float = 0.20   # 이름 + ID 둘 다 미등장
    heldout_item_ratio: float = 0.20
    heldout_id_only_ratio: float = 0.10    # 이름은 등장, ID만 미등장
    heldout_order_id_ratio: float = 0.10

    def item_name_capacity(self) -> dict[str, int]:
        """카테고리별로 만들 수 있는 서로 다른 상품명 개수."""
        if self.name_style == "single":
            return {c: len(CATEGORY_NOUNS[c]) for c in CATEGORIES}
        return {c: len(CATEGORY_NOUNS[c]) * len(CATEGORY_MODIFIERS[c]) for c in CATEGORIES}

    def validate(self) -> None:
        p: list[str] = []
        if self.name_style not in ("single", "two_word"):
            p.append("name_style 은 single 또는 two_word")
        if self.min_per_city * len(CITIES) > self.n_customers:
            p.append("city 최소 인원 합 > 고객 수")
        if self.min_per_membership * len(MEMBERSHIPS) > self.n_customers:
            p.append("membership 최소 인원 합 > 고객 수")
        if self.min_per_category * len(CATEGORIES) > self.n_items:
            p.append("category 최소 개수 합 > 상품 수")
        if self.min_per_quantity * len(self.quantity_values) > self.n_orders:
            p.append("quantity 최소 건수 합 > 주문 수")
        if self.min_per_status * len(STATUSES) > self.n_orders:
            p.append("status 최소 건수 합 > 주문 수")
        if self.n_customers > self.n_orders:
            p.append("주문 수 >= 고객 수 여야 모든 고객이 최소 1건")
        if self.n_items > self.n_orders:
            p.append("주문 수 >= 상품 수 여야 모든 상품이 최소 1건")
        if self.min_zero_stock > self.n_items:
            p.append("품절 상품 수 > 전체 상품 수")
        for nm, (lo, hi), need in [("customer_id", self.customer_id_range, self.n_customers),
                                   ("item_id", self.item_id_range, self.n_items),
                                   ("order_id", self.order_id_range, self.n_orders)]:
            if hi - lo + 1 < need:
                p.append(f"{nm} 대역이 좁아 고유값 {need}개 불가")
        if self.name_style == "single" and len(GIVEN_NAMES) < self.n_customers:
            p.append(f"이름 풀 {len(GIVEN_NAMES)} < 고객 수 {self.n_customers}")
        cap = self.item_name_capacity()
        if sum(cap.values()) < self.n_items:
            p.append(f"상품명 풀 총량 {sum(cap.values())} < 상품 수 {self.n_items}")
        if p:
            raise ValueError("설정 오류:\n  - " + "\n  - ".join(p))


PRESETS: dict[str, Config] = {
    # 계획서 5장 원안
    "small": Config(n_customers=80, n_items=40, n_orders=150,
                    customer_id_range=(100, 399), item_id_range=(400, 699),
                    order_id_range=(700, 999),
                    min_per_city=10, min_per_membership=15, min_per_category=5,
                    min_per_quantity=8, min_per_status=25, min_zero_stock=3),
    # 확장판 (기본)
    "large": Config(),
}


SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    city        TEXT    NOT NULL,
    membership  TEXT    NOT NULL
);

CREATE TABLE items (
    item_id   INTEGER PRIMARY KEY,
    item_name TEXT    NOT NULL UNIQUE,
    category  TEXT    NOT NULL,
    price     REAL    NOT NULL,
    stock     INTEGER NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    item_id     INTEGER NOT NULL REFERENCES items(item_id),
    quantity    INTEGER NOT NULL,
    status      TEXT    NOT NULL
);
"""

TABLE_COLUMNS = {
    "customers": ["customer_id", "name", "city", "membership"],
    "items": ["item_id", "item_name", "category", "price", "stock"],
    "orders": ["order_id", "customer_id", "item_id", "quantity", "status"],
}


# ---------------------------------------------------------------------------
# 생성 보조
# ---------------------------------------------------------------------------

def balanced_column(rng: random.Random, values: list, total: int, minimum: int,
                    caps: dict | None = None) -> list:
    """각 값이 최소 `minimum`번, 많아야 caps[값]번 나오는 길이 `total` 열."""
    counts = {v: minimum for v in values}
    if caps:
        for v in values:
            if minimum > caps.get(v, total):
                raise ValueError(f"'{v}' 최소치 {minimum} > 상한 {caps[v]}")
    remaining = total - minimum * len(values)
    if remaining < 0:
        raise ValueError("최소치 합이 total 을 초과")
    for _ in range(remaining):
        pool = [v for v in values
                if counts[v] < (caps.get(v, total) if caps else total)]
        if not pool:
            raise ValueError("상한에 막혀 total 을 채울 수 없음")
        counts[rng.choice(pool)] += 1
    col = [v for v in values for _ in range(counts[v])]
    rng.shuffle(col)
    return col


def sample_ids(rng: random.Random, lo: int, hi: int, n: int) -> list[int]:
    return sorted(rng.sample(range(lo, hi + 1), n))


# ---------------------------------------------------------------------------
# 테이블 생성
# ---------------------------------------------------------------------------

def build_customers(rng: random.Random, cfg: Config) -> list[tuple]:
    ids = sample_ids(rng, *cfg.customer_id_range, cfg.n_customers)
    if cfg.name_style == "single":
        names = rng.sample(GIVEN_NAMES, cfg.n_customers)
    else:
        names, used = [], set()
        while len(names) < cfg.n_customers:
            a, b = rng.choice(GIVEN_NAMES), rng.choice(GIVEN_NAMES)
            if a == b:
                continue
            full = f"{a} {b}"
            if full not in used:
                used.add(full)
                names.append(full)
    cities = balanced_column(rng, CITIES, cfg.n_customers, cfg.min_per_city)
    grades = balanced_column(rng, MEMBERSHIPS, cfg.n_customers, cfg.min_per_membership)
    return list(zip(ids, names, cities, grades))


def build_items(rng: random.Random, cfg: Config) -> list[tuple]:
    ids = sample_ids(rng, *cfg.item_id_range, cfg.n_items)
    cats = balanced_column(rng, CATEGORIES, cfg.n_items, cfg.min_per_category,
                           caps=cfg.item_name_capacity())

    names, used = [], set()
    for cat in cats:
        nouns = CATEGORY_NOUNS[cat]
        while True:
            name = (rng.choice(nouns) if cfg.name_style == "single"
                    else f"{rng.choice(CATEGORY_MODIFIERS[cat])} {rng.choice(nouns)}")
            if name not in used:
                used.add(name)
                names.append(name)
                break

    prices = [round(rng.uniform(cfg.price_min, cfg.price_max), 2) for _ in range(cfg.n_items)]
    stocks = [rng.randint(cfg.stock_min, cfg.stock_max) for _ in range(cfg.n_items)]
    for i in rng.sample(range(cfg.n_items), cfg.min_zero_stock):
        stocks[i] = 0
    return list(zip(ids, names, cats, prices, stocks))


def build_orders(rng: random.Random, cfg: Config,
                 customer_ids: list[int], item_ids: list[int]) -> list[tuple]:
    ids = sample_ids(rng, *cfg.order_id_range, cfg.n_orders)
    cust = list(customer_ids) + [rng.choice(customer_ids)
                                 for _ in range(cfg.n_orders - len(customer_ids))]
    item = list(item_ids) + [rng.choice(item_ids)
                             for _ in range(cfg.n_orders - len(item_ids))]
    rng.shuffle(cust)
    rng.shuffle(item)
    qty = balanced_column(rng, list(cfg.quantity_values), cfg.n_orders, cfg.min_per_quantity)
    sts = balanced_column(rng, STATUSES, cfg.n_orders, cfg.min_per_status)
    return list(zip(ids, cust, item, qty, sts))


# ---------------------------------------------------------------------------
# 미등장 값
# ---------------------------------------------------------------------------

def build_holdout(rng, cfg, customers, items, orders) -> dict:
    """학습 SQL에 절대 나오면 안 되는 리터럴.

    heldout_rows     : 이름과 ID 둘 다 미등장 (처음 보는 개체)
    heldout_ids_only : 이름은 학습에 등장, ID만 미등장 (숫자 복사만 따로 측정)
    """
    c_rows = rng.sample(range(len(customers)), round(cfg.n_customers * cfg.heldout_customer_ratio))
    i_rows = rng.sample(range(len(items)), round(cfg.n_items * cfg.heldout_item_ratio))
    c_seen = [k for k in range(len(customers)) if k not in set(c_rows)]
    i_seen = [k for k in range(len(items)) if k not in set(i_rows)]
    c_only = rng.sample(c_seen, round(cfg.n_customers * cfg.heldout_id_only_ratio))
    i_only = rng.sample(i_seen, round(cfg.n_items * cfg.heldout_id_only_ratio))
    o_ids = rng.sample([o[0] for o in orders], round(cfg.n_orders * cfg.heldout_order_id_ratio))

    return {
        "note": "SQL 생성기는 학습 데이터 생성 시 아래 리터럴을 전부 제외한다.",
        "customers": {
            "heldout_rows": sorted(({"customer_id": customers[k][0], "name": customers[k][1]}
                                    for k in c_rows), key=lambda d: d["customer_id"]),
            "heldout_ids_only": sorted(customers[k][0] for k in c_only),
        },
        "items": {
            "heldout_rows": sorted(({"item_id": items[k][0], "item_name": items[k][1]}
                                    for k in i_rows), key=lambda d: d["item_id"]),
            "heldout_ids_only": sorted(items[k][0] for k in i_only),
        },
        "orders": {"heldout_ids": sorted(o_ids)},
    }


def load_bans(ho: dict) -> dict[str, dict[str, set]]:
    """holdout.json 을 (테이블, 컬럼) -> 금지 리터럴 집합 으로 편다."""
    c_ids = ({r["customer_id"] for r in ho["customers"]["heldout_rows"]}
             | set(ho["customers"]["heldout_ids_only"]))
    i_ids = ({r["item_id"] for r in ho["items"]["heldout_rows"]}
             | set(ho["items"]["heldout_ids_only"]))
    return {
        "customers": {"customer_id": c_ids,
                      "name": {r["name"] for r in ho["customers"]["heldout_rows"]}},
        "items": {"item_id": i_ids,
                  "item_name": {r["item_name"] for r in ho["items"]["heldout_rows"]}},
        "orders": {"order_id": set(ho["orders"]["heldout_ids"]),
                   "customer_id": c_ids, "item_id": i_ids},
    }


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------

def build(seed: int, out_dir: str, cfg: Config | None = None) -> tuple[str, str]:
    cfg = cfg or Config()
    cfg.validate()
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    customers = build_customers(rng, cfg)
    items = build_items(rng, cfg)
    orders = build_orders(rng, cfg, [c[0] for c in customers], [i[0] for i in items])
    holdout = build_holdout(rng, cfg, customers, items, orders)

    db_path = os.path.join(out_dir, "shop.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    con.executemany("INSERT INTO items VALUES (?,?,?,?,?)", items)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    con.commit()
    con.close()

    holdout["seed"] = seed
    holdout["config"] = asdict(cfg)
    ho_path = os.path.join(out_dir, "holdout.json")
    with open(ho_path, "w", encoding="utf-8") as f:
        json.dump(holdout, f, ensure_ascii=False, indent=2)
    return db_path, ho_path


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def verify(db_path: str, cfg: Config | None = None) -> list[str]:
    cfg = cfg or Config()
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    q = lambda s: con.execute(s).fetchall()
    fails: list[str] = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    for t, exp in [("customers", cfg.n_customers), ("items", cfg.n_items),
                   ("orders", cfg.n_orders)]:
        n = q(f"SELECT COUNT(*) FROM {t}")[0][0]
        chk(n == exp, f"{t} 행 수 {n} != {exp}")

    for t, c, (lo, hi) in [("customers", "customer_id", cfg.customer_id_range),
                           ("items", "item_id", cfg.item_id_range),
                           ("orders", "order_id", cfg.order_id_range)]:
        mn, mx, nd, n = q(f"SELECT MIN({c}), MAX({c}), COUNT(DISTINCT {c}), COUNT(*) FROM {t}")[0]
        chk(lo <= mn and mx <= hi, f"{t}.{c} 대역 이탈 ({mn}~{mx} / 기대 {lo}~{hi})")
        chk(nd == n, f"{t}.{c} 중복")

    for t, c in [("customers", "name"), ("items", "item_name")]:
        nd, n = q(f"SELECT COUNT(DISTINCT {c}), COUNT(*) FROM {t}")[0]
        chk(nd == n, f"{t}.{c} 중복 ({nd}/{n})")

    if cfg.name_style == "single":
        for t, c in [("customers", "name"), ("items", "item_name")]:
            bad = q(f"SELECT COUNT(*) FROM {t} WHERE {c} LIKE '% %'")[0][0]
            chk(bad == 0, f"{t}.{c} 공백 포함 {bad}건 (한 단어 규칙 위반)")

    for t, c, vals, mn_ in [("customers", "city", CITIES, cfg.min_per_city),
                            ("customers", "membership", MEMBERSHIPS, cfg.min_per_membership),
                            ("items", "category", CATEGORIES, cfg.min_per_category),
                            ("orders", "status", STATUSES, cfg.min_per_status)]:
        cnt = dict(q(f"SELECT {c}, COUNT(*) FROM {t} GROUP BY {c}"))
        chk(set(cnt) == set(vals), f"{t}.{c} 값 종류 불일치: {sorted(cnt)}")
        for v in vals:
            chk(cnt.get(v, 0) >= mn_, f"{t}.{c}='{v}' {cnt.get(v, 0)}건 < 최소 {mn_}")

    qc = dict(q("SELECT quantity, COUNT(*) FROM orders GROUP BY quantity"))
    chk(set(qc) == set(cfg.quantity_values), f"quantity 값 종류 불일치: {sorted(qc)}")
    for v in cfg.quantity_values:
        chk(qc.get(v, 0) >= cfg.min_per_quantity,
            f"quantity={v} {qc.get(v, 0)}건 < 최소 {cfg.min_per_quantity}")

    chk(q(f"SELECT COUNT(*) FROM items WHERE price < {cfg.price_min} OR price > {cfg.price_max}")[0][0] == 0,
        "price 범위 이탈")
    chk(q("SELECT COUNT(*) FROM items WHERE ROUND(price,2) != price")[0][0] == 0,
        "price 소수점 2자리 초과")
    chk(q(f"SELECT COUNT(*) FROM items WHERE stock < {cfg.stock_min} OR stock > {cfg.stock_max}")[0][0] == 0,
        "stock 범위 이탈")
    z = q("SELECT COUNT(*) FROM items WHERE stock = 0")[0][0]
    chk(z >= cfg.min_zero_stock, f"품절 {z}개 < 최소 {cfg.min_zero_stock}")

    chk(q("SELECT COUNT(DISTINCT customer_id) FROM orders")[0][0] == cfg.n_customers,
        "주문 없는 고객 존재")
    chk(q("SELECT COUNT(DISTINCT item_id) FROM orders")[0][0] == cfg.n_items,
        "주문 없는 상품 존재")
    chk(len(q("PRAGMA foreign_key_check")) == 0, "FK 위반")

    for t, cols in [("customers", ["name", "city", "membership"]),
                    ("items", ["item_name", "category"]), ("orders", ["status"])]:
        for c in cols:
            bad = q(f"SELECT COUNT(*) FROM {t} WHERE {c} != LOWER({c})")[0][0]
            chk(bad == 0, f"{t}.{c} 대문자 {bad}건")

    chk(q("SELECT COUNT(*) FROM customers c JOIN items i ON c.customer_id = i.item_id")[0][0] == 0,
        "customer_id 와 item_id 대역이 겹침")

    con.close()
    return fails


def canonical_hash(db_path: str) -> str:
    """파일 바이트가 아니라 행 내용으로 결정성을 판정한다."""
    con = sqlite3.connect(db_path)
    parts = []
    for t in ("customers", "items", "orders"):
        rows = con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
        parts.append(json.dumps([t] + [list(r) for r in rows], sort_keys=True))
    con.close()
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# SQL 공간 계산
# ---------------------------------------------------------------------------

def sql_space(db_path: str, ho_path: str, detail: bool = True) -> tuple[int, int]:
    """1단계 문법(SELECT 단일 컬럼 또는 *, WHERE 단일 등호)으로 만들 수 있는
    고유 SQL 수를 미등장 리터럴 제외하고 센다."""
    con = sqlite3.connect(db_path)
    bans = load_bans(json.load(open(ho_path, encoding="utf-8")))
    total = id_part = 0
    if detail:
        print(f"{'테이블':<11}{'WHERE 컬럼':<14}{'가용 리터럴':>11}{'SQL':>9}")
        print("-" * 46)
    for t, cols in TABLE_COLUMNS.items():
        n_sel = len(cols) + 1                      # 각 컬럼 + '*'
        for c in cols:
            vals = {r[0] for r in con.execute(f"SELECT DISTINCT {c} FROM {t}")}
            vals -= bans[t].get(c, set())
            n = n_sel * len(vals)
            total += n
            if c.endswith("_id"):
                id_part += n
            if detail:
                print(f"{t:<11}{c:<14}{len(vals):>11}{n:>9}")
    con.close()
    if detail:
        print("-" * 46)
        print(f"{'합계':<36}{total:>10}")
        print(f"{'ID 조건 비중':<36}{id_part * 100 // total:>9}%")
        print(f"{'ID 아닌 SQL':<36}{total - id_part:>10}")
    return total, id_part


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

def test_determinism(out_dir: str, cfg: Config) -> list[str]:
    a, _ = build(0, os.path.join(out_dir, "_t_a"), cfg)
    b, _ = build(0, os.path.join(out_dir, "_t_b"), cfg)
    c, _ = build(1, os.path.join(out_dir, "_t_c"), cfg)
    ha, hb, hc = canonical_hash(a), canonical_hash(b), canonical_hash(c)
    f = []
    if ha != hb:
        f.append("같은 시드인데 결과가 다름")
    if ha == hc:
        f.append("다른 시드인데 결과가 같음")
    return f


def test_price_equality(db_path: str) -> list[str]:
    """round(x,2) 저장값이 SQL 리터럴과 '=' 로 매칭되는지 전수 확인."""
    con = sqlite3.connect(db_path)
    f = []
    for iid, price in con.execute("SELECT item_id, price FROM items"):
        lit = f"{price:.2f}"
        hit = con.execute(
            f"SELECT COUNT(*) FROM items WHERE item_id={iid} AND price={lit}").fetchone()[0]
        if hit != 1:
            f.append(f"item_id={iid}: {price!r} 인데 '= {lit}' 매칭 실패")
    con.close()
    return f


def test_holdout(ho_path: str) -> list[str]:
    ho = json.load(open(ho_path, encoding="utf-8"))
    f = []
    for t, k in [("customers", "customer_id"), ("items", "item_id")]:
        rows = {r[k] for r in ho[t]["heldout_rows"]}
        only = set(ho[t]["heldout_ids_only"])
        if rows & only:
            f.append(f"{t}: heldout_rows 와 heldout_ids_only 가 겹침")
        if not rows or not only:
            f.append(f"{t}: 미등장 값 집합이 비어 있음")
    return f


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------

def summary(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    print("=" * 62)
    for t in ("customers", "items", "orders"):
        print(f"{t:<12}{con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>5} 행")
    print("=" * 62)
    for label, sql in [
        ("city", "SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY 2 DESC"),
        ("membership", "SELECT membership, COUNT(*) FROM customers GROUP BY membership ORDER BY 2 DESC"),
        ("category", "SELECT category, COUNT(*) FROM items GROUP BY category ORDER BY 2 DESC"),
        ("status", "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY 2 DESC"),
    ]:
        print(f"{label:<12}" + "  ".join(f"{v}:{c}" for v, c in con.execute(sql)))
    mn, mx, av = con.execute("SELECT MIN(price), MAX(price), ROUND(AVG(price),2) FROM items").fetchone()
    print(f"{'price':<12}{mn} ~ {mx} (평균 {av})")
    mn, mx, z = con.execute("SELECT MIN(stock), MAX(stock), SUM(stock=0) FROM items").fetchone()
    print(f"{'stock':<12}{mn} ~ {mx} (품절 {z}개)")
    print("-" * 62)
    for t in ("customers", "items", "orders"):
        for r in con.execute(f"SELECT * FROM {t} LIMIT 3"):
            print(f"  {t:<10}", r)
    con.close()


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="1단계 DB 생성기")
    ap.add_argument("command", choices=["build", "verify", "test", "summary", "space"])
    ap.add_argument("--preset", default="large", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--name-style", default=None, choices=["single", "two_word"])
    args = ap.parse_args()

    cfg = Config(**asdict(PRESETS[args.preset]))
    if args.name_style:
        cfg.name_style = args.name_style
    db = os.path.join(args.out, "shop.db")
    ho = os.path.join(args.out, "holdout.json")

    if args.command == "build":
        db, ho = build(args.seed, args.out, cfg)
        fails = verify(db, cfg)
        print(f"생성  {db}\n      {ho}")
        print("검증 ", "통과" if not fails else "실패\n  - " + "\n  - ".join(fails))
        print("해시 ", canonical_hash(db)[:16])
        total, idp = sql_space(db, ho, detail=False)
        print(f"SQL   고유 {total}개 (ID 조건 {idp * 100 // total}%)")
        return 1 if fails else 0

    if args.command == "verify":
        fails = verify(db, cfg)
        print("통과" if not fails else "실패\n  - " + "\n  - ".join(fails))
        return 1 if fails else 0

    if args.command == "test":
        db, ho = build(0, args.out, cfg)
        for name, fails in [("분포 조건", verify(db, cfg)),
                            ("결정성", test_determinism(args.out, cfg)),
                            ("price 등호", test_price_equality(db)),
                            ("미등장 값", test_holdout(ho))]:
            print(f"[{'PASS' if not fails else 'FAIL'}] {name}")
            for x in fails:
                print(f"       {x}")
        return 0

    if args.command == "space":
        sql_space(db, ho)
        return 0

    summary(db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
