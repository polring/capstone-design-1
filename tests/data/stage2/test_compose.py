import json
import re
import random

import pytest

from src.data.stage2 import compose as c
from src.data.stage2.sql_ast import Cond, Query, canonical, to_dict

FRAMES = json.loads(c.FRAMES_PATH.read_text(encoding="utf-8"))
LEXICON = json.loads(c.LEXICON_PATH.read_text(encoding="utf-8"))
TRAIN_SQL = c.PATHS.sql_dir / "train.json"

pytestmark = pytest.mark.integration  # 저장소의 조합기 데이터와 SQL 파일을 읽는다


def rec(table, select, conds=(), conn=None, order=None, limit=None) -> dict:
    return to_dict(canonical(Query(table, select, tuple(conds), conn, order, limit)))


def _train_rows():
    if not TRAIN_SQL.exists():
        pytest.skip("data/stage2/sql/train.json 없음 (python -m src.data.stage2.sql_gen build)")
    return json.loads(TRAIN_SQL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items_rows():
    return random.Random(0).sample([r for r in _train_rows() if r["table"] == "items"], 2000)


@pytest.fixture(scope="module")
def all_rows():
    """테이블마다 1000개"""
    rows = _train_rows()
    return [r for t in ("customers", "items", "orders")
            for r in random.Random(0).sample([r for r in rows if r["table"] == t], 1000)]


def test_sql_compose_and_copy_literals(all_rows):
    """세 테이블 학습 SQL 이 모두 문장이 되고, 리터럴과 LIMIT 이 질문에 그대로 들어간다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    vkeys = c.value_keys(LEXICON)
    for r in all_rows:
        q, _ = comp.compose(r, c.rng_for(0, 0, r["sql"]))
        assert c.literal_problems(q, r, vkeys) == [], (q, r["sql"])
        assert q == q.lower() and "  " not in q and "{" not in q


def test_train_split_never_uses_eval_entries(items_rows):
    """train 조합기는 eval 표시된 틀·표현을 쓰지 않고, eval 조합기는 eval 틀을 쓴다"""
    train, ev = c.Composer(FRAMES, LEXICON, "train"), c.Composer(FRAMES, LEXICON, "eval")
    used_eval = False
    for r in items_rows[:500]:
        _, trace = train.compose(r, c.rng_for(0, 0, r["sql"]))
        assert all(e.split == "train" for e in trace)
        _, trace = ev.compose(r, c.rng_for(0, 0, r["sql"]))
        used_eval |= trace[0].split == "eval"
    assert used_eval


def test_deterministic_and_varies_by_epoch(items_rows):
    """같은 (seed, epoch, sql) 이면 같은 문장, epoch 가 바뀌면 대체로 다른 문장"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    differ = 0
    for r in items_rows[:200]:
        a = comp.compose(r, c.rng_for(0, 0, r["sql"]))[0]
        assert a == comp.compose(r, c.rng_for(0, 0, r["sql"]))[0]
        differ += a != comp.compose(r, c.rng_for(0, 1, r["sql"]))[0]
    assert differ > 150


def test_limit_one_is_singular():
    """LIMIT 1 이면 단수 명사 (the cheapest item)"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("items", "*", order=("price", "DESC"), limit=1)
    for ep in range(30):
        q, _ = comp.compose(r, c.rng_for(0, ep, r["sql"]))
        assert "items" not in q and "products" not in q, q


def test_sort_phrase_goes_to_sentence_end():
    """LIMIT 없는 정렬 표현은 문장 끝에 붙는다 (what are … called, sorted by …?)"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    sorts = [e.text("pl") for e in comp.orders["items.price ASC"]["sort"]]
    r = rec("items", "item_name", [Cond("stock", "<", 10)], order=("price", "ASC"))
    for ep in range(30):
        q, _ = comp.compose(r, c.rng_for(0, ep, r["sql"]))
        assert any(q.rstrip("?").endswith(s) for s in sorts), q


def test_existence_frames_for_select_eq_col():
    """SELECT 가 = 조건 컬럼이면 존재 확인 틀(L1)을 쓴다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("items", "item_name", [Cond("item_name", "=", "kayak")])
    assert c.role_of(r) == "exists"
    for ep in range(10):
        q, _ = comp.compose(r, c.rng_for(0, ep, r["sql"]))
        assert any(w in q for w in ("is there", "do we have", "exist", "confirm", "verify", "check if")) and "kayak" in q, q


def test_unknown_table_and_ordered_existence_rejected():
    """사전에 없는 테이블, 정렬이 붙은 존재 확인 SQL 은 조합하지 않는다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    lex = json.loads(json.dumps(LEXICON))
    del lex["nouns"]["customers"]
    with pytest.raises(c.ComposeError):
        c.Composer(FRAMES, lex, "train").compose(rec("customers", "*", [Cond("city", "=", "boston")]), random.Random(0))
    bad = {**rec("items", "category", [Cond("category", "=", "kitchen")]), "order": {"col": "stock", "dir": "ASC"}}
    with pytest.raises(c.ComposeError):
        comp.compose(bad, random.Random(0))


def test_literal_problems():
    """값 복사 검사: 숫자는 값으로 비교하고(20 = 20.00 = $20), 빠진 값과 LIMIT 을 잡는다"""
    r = rec("items", "*", [Cond("price", "<", 20.0)], order=("price", "ASC"), limit=3)
    assert c.literal_problems("the 3 cheapest items under $20", r) == []
    assert c.literal_problems("the cheapest items under 20.00", r) == ["LIMIT 3 없음"]
    assert c.literal_problems("the 3 cheapest items", r) == ["값 20.0 없음"]


def test_lexicon_conflicts():
    """같은 위치의 같은 표현이 서로 다른 SQL 조각에 있으면 충돌, 커밋된 사전은 충돌 없음"""
    assert c.lexicon_conflicts(LEXICON) == []
    bad = json.loads(json.dumps(LEXICON))
    bad["conds"]["items.price <="]["post"].append("under {$v}")
    assert any("under {$v}" in x for x in c.lexicon_conflicts(bad))


def test_lexicon_conflicts_scope():
    """테이블이 다르면 충돌이 아니고, 같은 테이블이면 값 목록이 겹치지 않는 범주형 컬럼끼리만 허용한다"""
    lex = json.loads(json.dumps(LEXICON))
    assert "in {v}" in lex["conds"]["items.category ="]["post"] and "in {v}" in lex["conds"]["customers.city ="]["post"]
    assert "{v}" in lex["conds"]["customers.city ="]["pre"] and "{v}" in lex["conds"]["customers.membership ="]["pre"]
    assert c.lexicon_conflicts(lex) == []
    lex["conds"]["orders.item_id ="]["post"].append("by customer {v}")  # orders 안에서 id 컬럼끼리는 값이 겹친다
    assert any("by customer {v}" in x for x in c.lexicon_conflicts(lex))


def test_frames_restricted_by_table():
    """tables 로 제한한 틀은 그 테이블에만 쓴다: do we carry 는 items, orders 식별 컬럼(order_id)은 name 으로 묻지 않는다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    cust = rec("customers", "name", [Cond("city", "=", "boston"), Cond("membership", "=", "gold")], "AND")
    orders = rec("orders", "order_id", [Cond("customer_id", "=", 1452), Cond("status", "=", "pending")], "AND")
    for ep in range(80):
        q = comp.compose(cust, c.rng_for(0, ep, cust["sql"]))[0]
        assert "carry" not in q and "catalog" not in q, q
        q = comp.compose(orders, c.rng_for(0, ep, orders["sql"]))[0]
        assert "name" not in q and "called" not in q and "carry" not in q, q


def test_unit_singular_after_one():
    """값 1 뒤 단위는 단수 (orders of 1 unit), 11·2.1 같은 값은 그대로"""
    assert c.finish("Orders of 1 units?") == "orders of 1 unit?"
    assert c.finish("orders of 11 units") == "orders of 11 units"
    assert c.finish("items under 2.1 dollars") == "items under 2.1 dollars"


def test_fixed_value_expressions():
    """L16: stock = 0 은 out of stock 류, stock > 0 은 in stock 류 표현도 쓰고, 다른 값에는 쓰지 않는다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    zero = rec("items", "*", [Cond("stock", "=", 0), Cond("category", "=", "kitchen")], "AND")
    some = rec("items", "*", [Cond("stock", ">", 0)])
    five = rec("items", "*", [Cond("stock", "=", 5)])
    qs = {name: [comp.compose(r, c.rng_for(0, ep, r["sql"]))[0] for ep in range(60)]
          for name, r in (("zero", zero), ("some", some), ("five", five))}
    assert any(("out of stock" in q or "sold out" in q or "none left" in q) for q in qs["zero"])
    assert any(("in stock" in q or "available" in q) and "0" not in q for q in qs["some"])
    assert not any("out of stock" in q or "sold out" in q for q in qs["five"])
    assert all("5" in q for q in qs["five"])


def test_multi_condition_existence_is_ident():
    """L14: 조건이 여러 개인 존재 확인 질문은 식별 컬럼 SQL 에만 붙고, = 조건 하나는 L1 로 남는다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    multi = rec("items", "item_name", [Cond("category", "=", "furniture"), Cond("price", "<", 40.0)], "AND")
    single = rec("items", "item_name", [Cond("category", "=", "furniture")])
    assert c.multi_exists_ok(multi) and not c.multi_exists_ok(single)
    asked = [comp.compose(multi, c.rng_for(0, ep, multi["sql"]))[0] for ep in range(80)]
    assert any(q.startswith(("do we have", "are there", "do we carry")) for q in asked)
    star = rec("items", "*", [Cond("category", "=", "furniture"), Cond("price", "<", 40.0)], "AND")
    for ep in range(40):
        q, _ = comp.compose(star, c.rng_for(0, ep, star["sql"]))
        assert not q.startswith(("do we have", "are there", "do we carry")), q


def test_tail_frames_only_for_short_noun_phrase():
    """명사구 뒤에 말이 남는 틀(… live?)은 조건 하나·정렬 없음일 때만 쓴다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    long = rec("customers", "city", [Cond("name", "LIKE", "ab%"), Cond("membership", "!=", "gold")], "AND")
    for ep in range(60):
        q, trace = comp.compose(long, c.rng_for(0, ep, long["sql"]))
        assert not c.frame_has_tail(trace[0].sg), q
    assert c.frame_has_tail("where do {np_def} live?") and not c.frame_has_tail("list the cities of {np_def}")


def test_no_stacked_with_and_short_like_quoted():
    """조건 표현 뒤에 'with the lowest …' 정렬 표현을 붙이지 않고, 3글자 이하 LIKE 조각은 따옴표로 감싼다(와일드카드 표기는 예외)"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("orders", "*", [Cond("quantity", "<", 9), Cond("status", "=", "shipped")], "AND",
            order=("quantity", "ASC"), limit=3)
    like = rec("customers", "*", [Cond("name", "LIKE", "%ed%")])
    for ep in range(60):
        q = comp.compose(r, c.rng_for(0, ep, r["sql"]))[0]
        assert "with the lowest" not in q and "with the smallest" not in q and "with the fewest" not in q, q
        q = comp.compose(like, c.rng_for(0, ep, like["sql"]))[0]
        assert "'ed'" in q or '"ed"' in q or re.search(r"[*%]ed[*%]", q), q


def test_compose_tagged():
    """태그 모드: 이름·ID·LIKE 조각·숫자는 질문 등장 순서로 val1, val2 … 가 되고 SQL 의 같은 자리에 들어간다.
    범주형 값은 그대로이고, 태그 없이 만든 문장과 틀·표현은 같다."""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("orders", "*", [Cond("customer_id", "=", 1452), Cond("quantity", ">=", 4), Cond("status", "=", "shipped")],
            "AND")
    for ep in range(30):
        q, sql, trace = comp.compose_tagged(r, c.rng_for(0, ep, r["sql"]))
        assert "1452" not in q and "shipped" in q and "'shipped'" in sql, (q, sql)
        assert {"val1", "val2"} <= set(re.findall(r"val\d", q)) and set(re.findall(r"val\d", sql)) == {"val1", "val2"}
        assert comp.compose(r, c.rng_for(0, ep, r["sql"]))[1] == trace
    like = rec("customers", "city", [Cond("name", "LIKE", "%ab%")])
    q, sql, _ = comp.compose_tagged(like, c.rng_for(0, 0, like["sql"]))
    assert "val1" in q and sql == "SELECT city FROM customers WHERE name LIKE '%val1%'"
    between = rec("items", "*", [Cond("price", "BETWEEN", 5.0, 30.0)])
    q, sql, _ = comp.compose_tagged(between, c.rng_for(0, 0, between["sql"]))
    assert sql == "SELECT * FROM items WHERE price BETWEEN val1 AND val2" and q.find("val1") < q.find("val2")


def test_compose_phrase_uses_one_eval_slot():
    """phrase 평가: train 조합에서 eval 자리 하나(틀 / 조건 키 / 정렬 키)에만 eval 표현이 들어가고 eval 명사는 쓰지 않는다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("items", "*", [Cond("category", "=", "kitchen"), Cond("price", ">", 20.0)], "AND", order=("price", "ASC"))
    fams = comp.eval_families(r)
    assert "frame" in fams and "cond:items.price >" in fams and "order:items.price ASC" in fams
    seen = set()
    for ep in range(30):
        q, _, trace, fam = comp.compose_phrase(r, 0, ep)
        evals = [e for e in trace if e.split == "eval"]
        assert evals and all(c.entry_family(e.ref) == fam for e in evals), (fam, [e.ref for e in evals])
        assert not any(e.ref.startswith("noun") for e in evals)
        seen.add(fam)
    assert len(seen) >= 2


SYNONYMS = json.loads(c.SYNONYMS_PATH.read_text(encoding="utf-8"))


def test_synonyms_change_frame_words_only():
    """유사어는 틀의 일상 단어만 바꾼다: {…} 자리와 그 안에 들어가는 표현·값은 그대로다"""
    syn = c.Synonyms({"prob": 1.0, "words": {"give me": ["get me"], "kitchen": ["cookware"]}})
    rng = random.Random(0)
    assert syn.apply("give me the prices of {np_def}", rng) == "get me the prices of {np_def}"
    assert syn.apply("{np_def} kitchenware", rng) == "{np_def} kitchenware"   # 단어 경계
    comp = c.Composer(FRAMES, LEXICON, "train", {**SYNONYMS, "prob": 1.0})
    r = rec("items", "*", [Cond("category", "=", "kitchen"), Cond("price", "<", 20.0)], "AND")
    qs = [comp.compose(r, c.rng_for(0, ep, r["sql"]))[0] for ep in range(60)]
    assert any("display" in q or "bring up" in q for q in qs)
    assert all("kitchen" in q and "20" in q for q in qs)


def test_synonym_problems():
    """커밋된 유사어 사전은 문제없고, 단서 단어를 넣거나 eval 전용 단어를 쓰는 대체어는 잡는다"""
    assert c.synonym_problems(SYNONYMS, FRAMES, LEXICON) == []
    bad = json.loads(json.dumps(SYNONYMS))
    bad["words"]["give me"].append("show me all")      # all 은 행 전체 단서
    bad["words"]["tell me"].append("got me")            # got 은 eval 전용 틀 단어
    probs = c.synonym_problems(bad, FRAMES, LEXICON)
    assert any("show me all" in p for p in probs) and any("got me" in p for p in probs)


def _many(comp, r, n=80):
    return [comp.compose(r, c.rng_for(0, ep, r["sql"]))[0] for ep in range(n)]


def test_noun_phrase_kinds_nodet_conds_head():
    """{np_nodet} 은 LIMIT 있는 SQL 에만(the 없이 "top 5 most expensive items"), {np_conds} 는 명사 없이 조건만
    ("order ids for customer 1102"), {np_head} 는 명사구 전체가 값일 때만("order 5312 status?") 쓴다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    lim = rec("items", "*", [Cond("category", "=", "kitchen")], order=("price", "DESC"), limit=5)
    assert any(q.startswith(("top 5", "5 ", "most", "priciest")) for q in _many(comp, lim))
    nolim = rec("items", "*", [Cond("category", "=", "kitchen")], order=("price", "DESC"))
    assert not any("{np_nodet}" in e.sg for e in comp.frames_for(nolim)[2])
    ids = rec("orders", "order_id", [Cond("customer_id", "=", 1102)])
    assert any(q.startswith(("order ids for customer 1102", "order numbers for customer 1102")) for q in _many(comp, ids))
    status = rec("orders", "status", [Cond("order_id", "=", 5312)])
    assert any(q.startswith(("order 5312 status", "order #5312 status")) for q in _many(comp, status))
    two = rec("orders", "status", [Cond("customer_id", "=", 1102), Cond("quantity", ">", 3)], "AND")
    assert not any("{np_head}" in e.sg for e in comp.frames_for(two)[2])


def test_category_value_as_noun_and_text_value_key():
    """범주 값이 명사 노릇을 한다 (electronics under $50, sports gear). 값 전용 키는 텍스트 값에도 쓴다 (still pending)"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("items", "*", [Cond("category", "=", "electronics"), Cond("price", "<", 50.0)], "AND")
    qs = _many(comp, r)
    assert any(not re.search(r"\b(items?|products?|things?)\b", q) for q in qs) and all("electronics" in q for q in qs)
    assert c.is_value_key("orders.status = pending") and not c.is_value_key("items.item_name LIKE prefix")
    pending = rec("orders", "*", [Cond("status", "=", "pending"), Cond("quantity", "<", 3)], "AND")
    shipped = rec("orders", "*", [Cond("status", "=", "shipped"), Cond("quantity", "<", 3)], "AND")
    assert any("still pending" in q or "waiting to ship" in q for q in _many(comp, pending))
    assert not any("still" in q or "waiting" in q for q in _many(comp, shipped))


def test_in_values_listed_in_any_order():
    """SQL 의 IN 값은 정규형(값 순서)이지만 질문에는 섞어 나열한다. 태그 모드는 섞지 않고 틀·표현은 태그 없는 문장과 같다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("customers", "city", [Cond("name", "IN", ("amy", "jane", "zoe"))])
    orders = set()
    for ep in range(40):
        q, trace = comp.compose(r, c.rng_for(0, ep, r["sql"]))
        pos = [re.search(rf"\b{n}\b", q).start() for n in ("amy", "jane", "zoe")]
        orders.add(tuple(sorted(range(3), key=pos.__getitem__)))
        tq, tsql, ttrace = comp.compose_tagged(r, c.rng_for(0, ep, r["sql"]))
        assert ttrace == trace and tq.find("val1") < tq.find("val2") < tq.find("val3")
    assert len(orders) >= 4


def test_question_order_sql():
    """L8: 학습 쌍의 SQL 은 IN 값을 질문에 나온 순서로 쓴다. AND·OR 조건 순서는 정규형 그대로, 값을 못 찾으면 실패"""
    r = rec("customers", "city", [Cond("name", "IN", ("amy", "jane", "zoe"))])
    assert (c.question_order_sql(r, "where do zoe, amy or jane live?")
            == "SELECT city FROM customers WHERE name IN ('zoe', 'amy', 'jane')")
    ids = rec("orders", "*", [Cond("customer_id", "IN", (1023, 1897)), Cond("status", "=", "pending")], "AND")
    assert (c.question_order_sql(ids, "pending orders of customers 1897 or 1023")
            == "SELECT * FROM orders WHERE customer_id IN (1897, 1023) AND status = 'pending'")
    a = rec("items", "*", [Cond("category", "=", "kitchen"), Cond("price", "<", 20.0)], "AND")
    assert c.question_order_sql(a, "items under $20 in kitchen") == a["sql"]
    o = rec("items", "*", [Cond("category", "=", "kitchen"), Cond("price", "<", 20.0)], "OR")
    assert c.question_order_sql(o, "items under $20 or in kitchen") == o["sql"]
    with pytest.raises(c.ComposeError):
        c.question_order_sql(r, "where do amy or jane live?")
    comp = c.Composer(FRAMES, LEXICON, "train")
    for ep in range(20):
        q, _ = comp.compose(r, c.rng_for(0, ep, r["sql"]))
        names = [x.split("'")[1] for x in c.question_order_sql(r, q).split(" OR ")]
        assert [q.index(n) for n in names] == sorted(q.index(n) for n in names)


def test_short_prefix_suffix_fragment_may_be_unquoted():
    """접두·접미의 짧은 조각은 따옴표 없이도 쓰고("starts with jo"), 조합기 문장의 단어와 같은 조각(a)은 늘 따옴표를 붙인다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    jo = rec("customers", "*", [Cond("name", "LIKE", "jo%")])
    qs = _many(comp, jo)
    assert any(re.search(r"(?<!['\"])\bjo\b(?!['\"])", q) for q in qs)
    assert any("'jo'" in q or '"jo"' in q for q in qs)
    assert "a" in comp.short_words
    a = rec("customers", "*", [Cond("name", "LIKE", "a%")])
    assert all("'a'" in q or '"a"' in q or re.search(r"\ba[*%]", q) for q in _many(comp, a))


def test_like_wildcard_unquoted():
    """와일드카드 표기(jo*, *ed, %ed%)는 방향을 기호가 정하므로 조각에 따옴표를 붙이지 않고, 방향이 SQL 과 맞는다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    for sql_val, pat in [("jo%", r"(?<![\w*%'\"])jo[*%]"), ("%ed", r"[*%]ed(?![\w*%'\"])"), ("%ed%", r"[*%]ed[*%]")]:
        qs = _many(comp, rec("customers", "*", [Cond("name", "LIKE", sql_val)]))
        wild = [q for q in qs if re.search(r"[*%]", q)]
        assert wild, sql_val
        assert all(re.search(pat, q) for q in wild), wild


def test_limit_tail_after_sort_phrase():
    """LIMIT 2 이상은 문장 끝 '정렬 표현, 개수'로도 쓴다 ("…, cheapest first, just 5")"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    sorts = [e.text("pl") for e in comp.orders["items.price ASC"]["sort"]]
    counts = [e.text("pl").replace("{n}", "5") for e in comp.limit_tails]
    r = rec("items", "item_name", [Cond("category", "=", "kitchen")], order=("price", "ASC"), limit=5)
    qs = _many(comp, r, 200)
    tails = [q for q in qs if any(q.rstrip("?").endswith(f"{s}, {n}") for s in sorts for n in counts)]
    assert tails and len(tails) < len(qs) / 2
    assert all(c.literal_problems(q, r) == [] for q in qs)


def test_sort_tail_without_comma_only_for_sorted_by():
    """명사구로 끝나는 틀에서 sorted / ordered / order by 정렬 표현은 쉼표 없이도 붙고, 다른 정렬 표현은 늘 쉼표 뒤다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    sorts = [e.text("pl") for e in comp.orders["items.price ASC"]["sort"] if e.split == "train"]
    r = rec("items", "*", [Cond("category", "=", "kitchen")], order=("price", "ASC"))
    qs = _many(comp, r, 200)
    bare = [q for q in qs if not any(q.rstrip("?").endswith(f", {s}") for s in sorts)]
    assert bare
    assert all(c.NO_COMMA_SORT.search(q) for q in bare), [q for q in bare if not c.NO_COMMA_SORT.search(q)]


def test_or_across_columns_joined_with_or_after_noun():
    """서로 다른 컬럼의 OR: 조건을 모두 명사 뒤에 두고 or 로 잇는다. 값 나열(IN)이 들어 있으면 쉼표로 경계를 둔다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("items", "*", [Cond("category", "=", "kitchen"), Cond("price", "<", 10.0)], "OR")
    for q in _many(comp, r):
        assert " or " in q and c.literal_problems(q, r) == [], q
        assert not re.search(r"\bkitchen (items|things|products|stuff)", q), q  # 명사 앞에 두지 않는다
    r = rec("items", "*", [Cond("category", "IN", ("kitchen", "sports")), Cond("price", "<", 10.0)], "OR")
    for q in _many(comp, r):
        assert ", or " in q and c.literal_problems(q, r) == [], q


def test_value_list_joins():
    """값 나열(IN)은 or 말고도 and / 쉼표 / 슬래시로 잇고, 공백 연결은 숫자 값에만 쓴다. 라벨은 늘 질문 순서"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    ids = rec("orders", "*", [Cond("customer_id", "IN", (1224, 1268))])
    cities = rec("customers", "*", [Cond("city", "IN", ("new york", "denver"))])
    qs = _many(comp, ids, 200)
    assert any(re.search(r"1224 and 1268|1268 and 1224", q) for q in qs)
    assert any(re.search(r"1224 1268|1268 1224", q) for q in qs)
    assert all(c.literal_problems(q, ids) == [] for q in qs)
    cq = _many(comp, cities, 200)
    assert any("/" in q for q in cq) and not any(re.search(r"new york denver|denver new york", q) for q in cq)
    assert all(c.question_order_sql(cities, q) for q in cq)


def test_limit_one_single_and_tail():
    """LIMIT 1 은 최상급 말고도 'the single …', 'the one … with the most …', 끝 표현('…, just the top one')으로 쓴다"""
    comp = c.Composer(FRAMES, LEXICON, "train")
    r = rec("orders", "order_id", [Cond("item_id", "=", 3281)], order=("quantity", "DESC"), limit=1)
    qs = _many(comp, r, 300)
    tails1 = [e.text("sg") for e in comp.limit_tails1]
    assert any(" single " in q or " the one " in q for q in qs)
    assert any(q.rstrip("?").endswith(tuple(tails1)) for q in qs)
