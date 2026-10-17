import json

import pytest

from src.data.stage2 import question_gen as qg
from src.data.stage2.sql_ast import Cond, Query, canonical, to_dict

pytestmark = pytest.mark.unit


def rec(table, select, conds=(), conn=None, order=None, limit=None, **meta) -> dict:
    return dict(to_dict(canonical(Query(table, select, tuple(conds), conn, order, limit))), **meta)


IN_REC = rec("customers", "*", [Cond("city", "IN", ("boston", "denver")), Cond("membership", "=", "gold")], "AND")
LIKE_REC = rec("items", "item_name", [Cond("item_name", "LIKE", "jo%"), Cond("price", "<", 20.0)], "AND",
               ("price", "ASC"), 3)


def test_prompt_has_sql_personas_and_no_composer_text():
    """프롬프트에는 SQL 과 페르소나가 들어가고, 조합기 표현(예: 정렬 뒤 개수 'just {n}')은 들어가지 않는다"""
    p = qg.build_batch_prompt([(7, IN_REC), (9, LIKE_REC)])
    assert f"- id=7 | {IN_REC['sql']}" in p and f"- id=9 | {LIKE_REC['sql']}" in p
    assert all(name in p for name in qg.PERSONAS)
    assert "just {n}" not in p and "sorted by" not in p


def test_literal_and_label_checks():
    """값·LIKE 조각·LIMIT 이 다 있어야 통과하고, 라벨은 IN 값을 질문 순서로 쓴다. 대소문자는 보지 않는다"""
    assert qg.check_question("Gold members in Denver or Boston?", IN_REC) == []
    assert any("boston" in f for f in qg.check_question("gold members in denver", IN_REC))
    assert qg.check_question("the 3 cheapest items starting with jo under $20", LIKE_REC) == []
    assert any("LIMIT" in f for f in qg.check_question("cheapest items starting with jo under $20", LIKE_REC))
    assert qg.check_question("the three cheapest items starting with jo under $20 3", LIKE_REC) == []  # 숫자 단어는 flag 만
    assert any("숫자" in f for f in qg.soft_flags("the three cheapest items starting with jo under $20 3", LIKE_REC))
    out_of_stock = rec("items", "*", [Cond("stock", "=", 0)])
    assert qg.check_question("items that are out of stock", out_of_stock) == []  # L16 고정값
    assert any("밑줄" in f for f in qg.check_question("item_name of the 3 cheapest starting with jo under 20", LIKE_REC))


def test_make_handout_and_collect(tmp_path):
    """배포 묶음: 학습·평가 SQL 을 섞어 묶음 id 로 쓰고(파일럿은 학습 SQL, 겹치지 않음, 이전 라운드 SQL 제외),
    돌려받은 응답은 원래 id 로 라운드별 폴더에 나뉜다"""
    rows = [rec("items", "*", [Cond("price", "<", float(p))]) for p in range(10, 40)]
    sql = tmp_path / "train.json"
    sql.write_text(json.dumps(rows), encoding="utf-8")
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / qg.ROUND_SQL).write_text(json.dumps({0: rows[0]}), encoding="utf-8")
    ext = tmp_path / "ext_r1_indist"
    ext.mkdir()
    ev = {100 + k: dict(rec("customers", "*", [Cond("city", "=", c)]), split="indist", batch=1)
          for k, c in enumerate(["boston", "denver", "chicago", "seattle"])}
    (ext / qg.ROUND_SQL).write_text(json.dumps(ev), encoding="utf-8")
    out = tmp_path / "round"
    qg.make_handout(str(out), n=20, pilot_n=5, batch_size=8, seed=0, sql_path=str(sql), exclude_dirs=[str(prev)],
                    eval_counts={"indist": 3}, eval_from=str(tmp_path / "ext_r1"), eval_prefix=str(tmp_path / "ext_r2"))
    main = json.loads((out / qg.ROUND_SQL).read_text(encoding="utf-8"))
    pilot = json.loads((out / "pilot" / qg.ROUND_SQL).read_text(encoding="utf-8"))
    evr = json.loads((tmp_path / "ext_r2_indist" / qg.ROUND_SQL).read_text(encoding="utf-8"))
    assert len(main) == 20 and len(pilot) == 5 and not set(main) & set(pilot) and "0" not in main and "0" not in pilot
    assert len(evr) == 3 and set(evr) <= {str(k) for k in ev}
    assert json.loads((tmp_path / "ext_r2_indist" / qg.ROUND_META).read_text(encoding="utf-8"))["split"] == "indist"
    readme = (out / "handout" / "README.md").read_text(encoding="utf-8")
    assert "<<" not in readme and "batch_03.txt" in readme  # 23개 / 8 = 배치 3개
    hmap = json.loads((out / qg.HANDOUT_MAP).read_text(encoding="utf-8"))
    assert sorted(map(int, hmap)) == list(range(1, 29))

    # 플랫폼 둘이 배치를 나눠 맡는다: batch_01 은 gpt, 나머지는 gemini
    rets = {plat: tmp_path / f"returned_{plat}" for plat in ("gpt", "gemini")}
    for f in sorted((out / "handout" / "sql").glob("batch_*.txt")):
        items = []
        for line in f.read_text(encoding="utf-8").splitlines():
            hid, s = line.removeprefix("id=").split(" | ")
            m = hmap[hid]
            r = (evr if m["round"].endswith("ext_r2_indist") else main)[str(m["id"])]
            assert r["sql"] == s
            v = f"{r['conds'][0]['val']:.2f}" if r["table"] == "items" else r["conds"][0]["val"]
            items.append({"id": int(hid), "questions": [f"{w} {v}" for w in ("a", "b", "c", "d", "e")]})
        d = rets["gpt" if f.stem == "batch_01" else "gemini"] / "responses"
        d.mkdir(parents=True, exist_ok=True)
        text = json.dumps(items)
        if f.stem == "batch_02":
            text = f"```json\n{text}\n```"  # 코드 블록으로 감싸 온 응답도 받는다
        (d / f"{f.stem}.json").write_text(text, encoding="utf-8")
    for plat, d in rets.items():
        qg.collect_handout(str(out), plat, str(d))
    back = {i["id"] for plat in rets for f in (tmp_path / "ext_r2_indist" / "responses" / plat).glob("*.json")
            for i in json.loads(f.read_text(encoding="utf-8"))}
    assert back == {int(k) for k in evr}
    # 플랫폼마다 맡은 배치의 SQL 만 보므로 '응답 없음' 이 없고, 둘을 합치면 학습 SQL 전부다
    report = json.loads((out / qg.REPORT).read_text(encoding="utf-8"))["platforms"]
    assert all("응답 없음" not in r["failure_reasons"] for r in report.values())
    assert sum(r["n_sql"] for r in report.values()) == 20


def test_make_handout_fixed_prompt_eval_only(tmp_path):
    """프롬프트를 정해 주는 평가 전용 묶음: 배치 파일마다 프롬프트 + SQL 이 들어가고, 학습 라운드와 파일럿은 만들지 않는다"""
    ext = tmp_path / "ext_r2_indist"
    ext.mkdir()
    ev = {100 + k: dict(rec("customers", "*", [Cond("city", "=", c)]), split="indist", batch=1)
          for k, c in enumerate(["boston", "denver", "chicago", "seattle", "houston"])}
    (ext / qg.ROUND_SQL).write_text(json.dumps(ev), encoding="utf-8")
    prompt = tmp_path / "prompt_final.txt"
    prompt.write_text("Write questions.\n", encoding="utf-8")
    out = tmp_path / "handout_round"
    qg.make_handout(str(out), n=0, pilot_n=25, batch_size=2, seed=0, eval_counts={"indist": 5},
                    eval_from=str(tmp_path / "ext_r2"), eval_prefix=str(tmp_path / "ext_r3"), prompt=str(prompt))
    assert not (out / qg.ROUND_SQL).exists() and not (out / "pilot").exists()
    assert not (out / "handout" / "sql" / "pilot.txt").exists()
    evr = json.loads((tmp_path / "ext_r3_indist" / qg.ROUND_SQL).read_text(encoding="utf-8"))
    assert set(evr) == {str(k) for k in ev}
    batches = sorted((out / "handout" / "sql").glob("batch_*.txt"))
    assert len(batches) == 3
    for f in batches:
        head, body = f.read_text(encoding="utf-8").split("\n\n", 1)
        assert head == "Write questions." and all(line.startswith("id=") for line in body.splitlines())
    readme = (out / "handout" / "README.md").read_text(encoding="utf-8")
    assert "<<" not in readme and "batch_03.txt" in readme


def test_validate_round(tmp_path):
    """플랫폼 폴더마다 쌍·통과율을 내고, 같은 질문이 다른 SQL 에 붙으면 양쪽 다 뺀다"""
    # SELECT 만 다른 SQL: 값 검사는 둘 다 통과하는 문장이 생긴다
    other = rec("customers", "membership", [Cond("city", "IN", ("boston", "denver")), Cond("membership", "=", "gold")], "AND")
    meta = {1: dict(IN_REC, split="indist", batch=1), 2: dict(other, split="indist", batch=1)}
    (tmp_path / qg.ROUND_SQL).write_text(json.dumps(meta), encoding="utf-8")
    good = ["gold denver boston members", "full records of gold ppl in boston or denver", "members gold in denver and boston",
            "could you list the gold members living in boston or denver for our mailing", "need gold members, denver/boston"]
    resp = [{"id": 1, "questions": good},
            {"id": 2, "questions": ["names gold boston denver", "full records of gold ppl in boston or denver", "x", "y", "z"]}]
    (tmp_path / "responses" / "gpt").mkdir(parents=True)
    (tmp_path / "responses" / "gpt" / "batch_01.json").write_text(json.dumps(resp), encoding="utf-8")
    report = qg.validate(str(tmp_path))
    pairs = json.loads((tmp_path / "pairs_gpt.json").read_text(encoding="utf-8"))
    qs = {p["question"] for p in pairs}
    assert "full records of gold ppl in boston or denver" not in qs  # 다른 SQL 과 같은 질문
    assert "names gold boston denver" in qs and "x" not in qs
    p = next(p for p in pairs if p["question"] == "gold denver boston members")
    assert p["sql"] == ("SELECT * FROM customers WHERE city IN ('denver', 'boston') AND membership = 'gold'")
    assert p["sql_group"] == IN_REC["sql"] and p["persona"] == "keyword" and p["platform"] == "gpt"
    assert report["platforms"]["gpt"]["pass_rate"]["keyword"] == 1.0
