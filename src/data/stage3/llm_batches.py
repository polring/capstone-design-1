# 역할: 고정된 정답 SQL별 LLM 질문 생성 배치를 준비·실행·수집하고 검증한다.
# 사용처: 3단계 질문 생성 라운드에서 독립 배치를 관리하고 결과를 검사할 때 사용한다.

"""Prepare/collect isolated question-generation batches with immutable, code-normalized SQL."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import shutil
import sqlite3
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from decimal import Decimal
import math
from src.config import stage_paths

from src.data.stage3 import sql_ast

PATHS = stage_paths(3)


def round_config(out):
    path = Path(out) / 'round_meta.json'
    meta = read(path) if path.exists() else {}
    n = meta.get('sql_count', len(read(Path(out) / 'round_sql.json')))
    batch_size = meta.get('batch_size', 25)
    questions = meta.get('questions_per_sql', 5)
    return n, batch_size, questions, math.ceil(n / batch_size)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized(record):
    query = sql_ast.from_dict(record)
    if not isinstance(query, sql_ast.AdvancedQuery):
        query = replace(query, policy_order=True, tie_break=None)
    return sql_ast.to_dict(query)


def prepare(args):
    out = Path(args.out)
    if (out / "round_sql.json").exists():
        raise FileExistsError("Round exists; use run or collect, or choose another output directory")
    rng = random.Random(args.seed)
    excluded = {normalized(r)["sql"] for p in args.exclude for r in read(p)}
    evaluation = {normalized(r)["sql"] for r in read(args.eval_sql)}
    used = excluded | evaluation
    selected = []
    n_sql, batch_size, questions = args.n_sql, args.batch_size, args.questions_per_sql
    families = ['legacy', 'aggregate', 'group', 'having', 'scalar_subquery']
    weights = [250, 225, 188, 187, 150]
    quotas = [n_sql * weight // 1000 for weight in weights]
    for index in sorted(range(len(weights)), key=lambda i: (-(n_sql * weights[i] % 1000), i))[:n_sql - sum(quotas)]:
        quotas[index] += 1
    groups = [("legacy", [r for r in read(args.legacy_sql) if not r.get("query_kind")], quotas[0])]
    advanced = [r for r in read(args.advanced_sql) if r.get("query_kind")]
    for family, n in zip(families[1:], quotas[1:]):
        groups.append((family, [r for r in advanced if r["family"] == family], n))
    for family, source, n in groups:
        source = read(source) if isinstance(source, str) else source
        eligible = []
        for record in source:
            row = normalized(record)
            if row["sql"] not in used:
                eligible.append({**row, "source_family": family, "source_sql": record["sql"]})
        chosen = rng.sample(eligible, n)
        selected += chosen
        used.update(r["sql"] for r in chosen)
    rng.shuffle(selected)
    assert len(selected) == len({r['sql'] for r in selected}) == n_sql
    con = sqlite3.connect(f"file:{Path(args.db).resolve().as_posix()}?mode=ro", uri=True)
    for r in selected:
        con.execute(r["sql"]).fetchall()
        assert not sql_ast.violations(sql_ast.parse(r["sql"]))
    con.close()
    save(out / "round_sql.json", selected)
    schema = {"type": "object", "properties": {"records": {"type": "array", "minItems":1,"maxItems":batch_size,"items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "sql": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}, "minItems": questions, "maxItems": questions}},
        "required": ["id", "sql", "questions"], "additionalProperties": False}}},
        "required": ["records"], "additionalProperties": False}
    save(out / "schema.json", schema)
    brief = Path(args.brief).read_text(encoding="utf-8")
    brief = brief.replace("{questions_per_sql}", str(questions)).replace("{batch_size}", str(batch_size))
    brief += f"\nWrite exactly {questions} distinct questions per SQL. Prioritize common direct questions and short requests, with varied conversational, imperative and concise contextual wording. Avoid unnecessary database jargon. Keep all values exact.\n"
    sessions = math.ceil(n_sql / batch_size)
    for batch in range(sessions):
        records = [{"id": i+1, "sql": r["sql"]} for i,r in enumerate(selected) if i//batch_size == batch]
        path = out / "prompts" / f"batch_{batch+1:02}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(brief + "\nINPUT:\n" + json.dumps(records, ensure_ascii=False), encoding="utf-8")
    save(out / "round_meta.json", {"seed": args.seed, "split": "train", "sql_count": n_sql,
        "questions_per_sql": questions, "expected_pairs": n_sql * questions, "batch_size": batch_size, "sessions": sessions,
        "families": dict(Counter(r['source_family'] for r in selected)), "policy": "2026-10-15",
        "excluded_previous_sql": len(excluded), "eval_overlap": 0,
        "brief_sha256": hashlib.sha256(Path(args.brief).read_bytes()).hexdigest()})
    print(f"Prepared {sessions} independent batches: {n_sql} SQL -> {n_sql * questions} questions", flush=True)


def validate_batch(out, batch):
    source=read(out/'round_sql.json')
    rows=read(out/'responses'/f'batch_{batch:02}.json')['records']
    n, batch_size, questions, sessions = round_config(out)
    assert 1 <= batch <= sessions, 'Invalid batch number'
    expected=set(range((batch-1)*batch_size+1,min(batch*batch_size,n)+1))
    assert len(rows)==len(expected) and {r['id'] for r in rows}==expected, 'Missing or duplicate SQL IDs'
    for row in rows:
        assert row['sql']==source[row['id']-1]['sql'], 'SQL changed by LLM'
        assert len(row['questions'])==len(set(row['questions']))==questions, 'Need five distinct questions' if questions == 5 else f'Need {questions} distinct questions'
        assert all(isinstance(q,str) and q.strip() for q in row['questions']), 'Empty question'


def run_one(out, batch):
    name = f"batch_{batch:02}"
    dest = out / "responses" / f"{name}.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        validate_batch(out,batch)
        return {"batch": batch, "status": "existing"}
    session = out / "session"
    session.mkdir(exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    command = [shutil.which("codex"), "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
        "-C", str(session.resolve()), "--output-schema", str((out / "schema.json").resolve()),
        "--output-last-message", str(dest.resolve()), "--color", "never", "-"]
    with (logs / f"{name}.txt").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, input=(out/"prompts"/f"{name}.txt").read_text(encoding="utf-8"),
                                text=True, encoding="utf-8", stdout=log, stderr=log)
    status = {"batch": batch, "status": "returned" if result.returncode == 0 and dest.exists() else "failed",
              "exit_code": result.returncode}
    if status['status']=='returned':validate_batch(out,batch)
    print(json.dumps(status), flush=True)
    return status


def run(args):
    out = Path(args.out)
    _, _, _, sessions = round_config(out)
    missing = [b for b in range(1,sessions+1) if not (out/"responses"/f"batch_{b:02}.json").exists()]
    if not missing:
        return
    # Check availability before starting 39 more calls when a quota is exhausted.
    first = run_one(out, missing.pop(0))
    if first["status"] == "failed":
        save(out/"execution_status.json", {"complete": False, "results": [first]})
        raise RuntimeError("First batch failed; see log. Remaining batches were not launched.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [first] + list(pool.map(lambda b: run_one(out,b), missing))
    save(out/"execution_status.json", {"complete": all(x['status']!='failed' for x in results), "results": results})
    if any(x['status']=='failed' for x in results):
        raise RuntimeError("Incomplete batches; see execution_status.json")


def collect(args):
    out = Path(args.out)
    sql = read(out/"round_sql.json")
    pairs, errors, warnings, session_ids = [], [], [], []
    n, batch_size, questions, sessions = round_config(out)
    for batch in range(1,sessions+1):
        path = out/"responses"/f"batch_{batch:02}.json"
        if not path.exists():
            errors.append(f"missing batch {batch}")
            continue
        records = read(path)["records"]
        expected = set(range((batch-1)*batch_size+1,min(batch*batch_size,n)+1))
        assert len(records)==len(expected) and {r['id'] for r in records}==expected
        for r in sorted(records,key=lambda r:r['id']):
            assert r['sql']==sql[r['id']-1]['sql'], "SQL changed by LLM"
            assert len(r['questions'])==len(set(r['questions']))==questions
            for style,q in enumerate(r['questions']):
                assert isinstance(q,str) and q.strip()
                literals = re.findall(r"'([^']*)'",r['sql'])
                for value in literals:
                    if value.strip('%').lower() not in q.lower():
                        warnings.append({"sql_id":r['id'],"style":style,"missing_string":value})
                pairs.append({"question":q,"sql":r['sql'],"sql_group":r['sql'],
                    "source":"llm:gpt:independent_session","round":out.name,"batch":batch,"style":style,
                    "sql_id":r['id'],"family":sql[r['id']-1]['source_family']})
        log = out/"logs"/f"batch_{batch:02}.txt"
        if log.exists():session_ids += re.findall(r"session id:\s*(\S+)",log.read_text(encoding='utf-8',errors='replace'))
    labels = {}
    conflicts = []
    for p in pairs:
        key = ' '.join(p['question'].lower().split())
        if key in labels and labels[key]!=p['sql']:conflicts.append(key)
        labels[key] = p['sql']
    save(out/"pairs_candidate.json",pairs)
    report = {"pairs":len(pairs),"unique_sql":len({p['sql'] for p in pairs}),"errors":errors,
        "literal_warnings":warnings,"label_conflicts":conflicts,"session_ids":session_ids,
        "semantic_review":"structural/value checks only; family/template audit and sample review required",
        "promoted_to_training":False}
    save(out/"validation_report.json",report)
    print(f"Collected {len(pairs)} pairs, errors={len(errors)}, string warnings={len(warnings)}, conflicts={len(conflicts)}",flush=True)


def audit(args):
    """Cheap whole-round checks plus stratified semantic-review samples, not per-pair GPT review."""
    from src.data import db_gen
    from src.data.stage3.clean_ambiguous import judge
    from src.tokenizer.bpe import Tokenizer
    out=Path(args.out)
    n_sql, _, questions_per_sql, _ = round_config(out)
    pair_path=Path(args.pairs) if getattr(args,'pairs',None) else out/'pairs_candidate.json'
    pairs=read(pair_path);rows=read(out/'round_sql.json')
    bans=db_gen.load_bans(read(args.holdout))
    evaluation={normalized(r)['sql'] for r in read(args.eval_sql)}
    previous={normalized(r)['sql'] for path in args.exclude for r in read(path)}
    errors=[];flags=[];reviews=[];lengths=[];sql_errors=[]
    con=sqlite3.connect(f"file:{Path(args.db).resolve().as_posix()}?mode=ro",uri=True)
    for index,row in enumerate(rows,1):
        query=sql_ast.parse(row['sql'])
        if sql_ast.violations(query):sql_errors.append({'sql_id':index,'reason':'AST validation'})
        if row['sql'] in evaluation or row['sql'] in previous:sql_errors.append({'sql_id':index,'reason':'eval/previous round overlap'})
        conditions=(*query.conds,*(query.inner_conds if isinstance(query,sql_ast.AdvancedQuery) else ()))
        for c in conditions:
            values=c.val if c.op=='IN' else (c.val,c.val2) if c.op=='BETWEEN' else (c.val,)
            if any(v in bans[query.table].get(c.col,set()) for v in values):sql_errors.append({'sql_id':index,'reason':'holdout literal leak'})
        try:con.execute(row['sql']).fetchall()
        except sqlite3.Error as exc:sql_errors.append({'sql_id':index,'reason':str(exc)})
    con.close()
    tok=Tokenizer.load(args.tokenizer)
    sql_tokens={row['sql']:len(tok.encode(row['sql'])) for row in rows}
    cell={};labels={}
    for i,pair in enumerate(pairs,1):
        row=rows[pair['sql_id']-1]
        if pair['sql']!=row['sql']:errors.append({'pair_id':i,'reason':'SQL mutation'})
        judgment=judge(pair)
        if judgment.status=='violation':
            flags.append({'pair_id':i,'reason':judgment.reason,'evidence':judgment.evidence,
                          'question':pair['question'],'sql':pair['sql']})
        elif judgment.status=='review':
            reviews.append({'pair_id':i,'evidence':judgment.evidence,'unchecked':judgment.unchecked,
                            'question':pair['question'],'sql':pair['sql']})
        stripped=re.sub(r"'[^']*'",'',pair['sql'])
        nums={Decimal(n) for n in re.findall(r'(?<![\w])\d+(?:\.\d+)?',stripped)}
        qnums={Decimal(n) for n in re.findall(r'(?<![\w])\d+(?:\.\d+)?',pair['question'])}
        missing=nums-qnums-{Decimal(0),Decimal(1)}
        query=sql_ast.parse(pair['sql'])
        all_conditions=(*query.conds,*(query.inner_conds if isinstance(query,sql_ast.AdvancedQuery) else ()))
        if query.table=='items' and re.search(r'\blow[- ]stock\b|\brunning low on stock\b',pair['question'],re.I):
            if any(c.col=='stock' and c.op=='<=' and c.val==10 for c in all_conditions):
                missing.discard(Decimal(10))  # L20 fixed default, not a dropped literal.
        if missing:flags.append({'pair_id':i,'reason':'missing numeric values: '+str(sorted(map(str,missing))),'question':pair['question'],'sql':pair['sql']})
        for literal in re.findall(r"'([^']*)'",pair['sql']):
            if literal.strip('%').lower() not in pair['question'].lower():
                flags.append({'pair_id':i,'reason':'missing literal '+literal,'question':pair['question'],'sql':pair['sql']})
        length=len(tok.encode(pair['question']))+sql_tokens[pair['sql']]+3;lengths.append(length)
        if length>256:flags.append({'pair_id':i,'reason':f'context length {length}>256 with current tokenizer','question':pair['question'],'sql':pair['sql']})
        key=' '.join(pair['question'].lower().split())
        if key in labels and labels[key]!=pair['sql']:errors.append({'pair_id':i,'reason':'question label conflict'})
        labels[key]=pair['sql']
        cell.setdefault((pair['family'],pair['style']),[]).append((i,pair))
    rng=random.Random(20261015);samples=[]
    for key,group in sorted(cell.items()):
        for i,pair in rng.sample(group,min(4,len(group))):samples.append({'pair_id':i,**pair})
    save(out/'stratified_sample.json',samples)
    save(out/'quality_flags.json',flags)
    save(out/'review_flags.json',reviews)
    report={'pairs_file':str(pair_path),'expected_pairs':n_sql * questions_per_sql,'returned_pairs':len(pairs),'unique_sql':len({p['sql'] for p in pairs}),
        'family_pairs':dict(Counter(p['family'] for p in pairs)),'errors':errors,'sql_errors':sql_errors,
        'quality_flag_count':len(flags),'quality_flagged_pairs':len({p['pair_id'] for p in flags}),
        'review_flag_count':len(reviews),'review_flagged_pairs':len({p['pair_id'] for p in reviews}),
        'max_tokens_current_tokenizer':max(lengths,default=0),'sample_count':len(samples),
        'sampling_cells':len(cell),'sample_semantic_audit':'pending','promoted_to_training':False}
    save(out/'quality_report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    assert len(pairs)==n_sql * questions_per_sql and len(rows)==n_sql and not errors and not sql_errors
    # 라벨 규칙 위반·값 누락·길이 초과 경고가 있으면 실패로 끝낸다. 경고 쌍은 확인 전까지 학습 폴더로 옮기지 않는다
    if flags or reviews:
        print(f"품질 경고 {len(flags)}건 ({len({f['pair_id'] for f in flags})}쌍): {out/'quality_flags.json'} 확인 전 학습 편입 금지",flush=True)
        if reviews:
            print(f"검토 필요 {len(reviews)}쌍: {out/'review_flags.json'} 확인 전 학습 편입 금지",flush=True)
        return 1
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','collect','audit','all'])
    p.add_argument('--out',required=True,help='독립 질문 생성 라운드 폴더')
    p.add_argument('--n-sql',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=25)
    p.add_argument('--questions-per-sql',type=int,default=5)
    p.add_argument('--pairs',help='Audit a corrected candidate file while retaining the original responses')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--legacy-sql', '--sql', dest='legacy_sql',default=str(PATHS.sql_dir / 'train.json'))
    p.add_argument('--advanced-sql',default=str(PATHS.sql_dir / 'train.json'))
    p.add_argument('--eval-sql',default=str(PATHS.eval_dir / 'sql.json'))
    p.add_argument('--db',default=str(PATHS.db_path))
    p.add_argument('--brief',default=str(PATHS.raw_dir / 'composer/llm_brief.txt'))
    p.add_argument('--holdout',default=str(PATHS.holdout_path))
    p.add_argument('--tokenizer',default=str(PATHS.tokenizer_path))
    p.add_argument('--exclude',nargs='*',default=[],help='제외할 이전 라운드 SQL 파일')
    args=p.parse_args(argv)
    if min(args.n_sql,args.batch_size,args.questions_per_sql,args.workers) <= 0:
        p.error('SQL 수·배치 크기·질문 수·worker 수는 양수여야 합니다')
    if args.command in ('prepare','all'):prepare(args)
    if args.command in ('run','all'):run(args)
    if args.command in ('collect','all'):collect(args)
    if args.command in ('audit','all'):return audit(args)
    return 0


if __name__=='__main__':raise SystemExit(main())
