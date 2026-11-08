# 역할: 재고 관리 도메인의 단어·구문·약어·기호 치환 규칙을 사전으로 만든다.
# 사용처: 3단계 normalizer.json을 생성해 Python·C 입력 전처리기에 제공할 때 사용한다.

"""Shop-domain noun maps and explicit phrase normalization, without model changes.

Human-defined schema sense anchors and phrase rules are external domain knowledge.
No evaluation SQL/questions are loaded by this builder; targets must be in train.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import zipfile
from collections import deque
from pathlib import Path

from src.data.preprocessing.wordnet_map import WordNetGraph, train_vocabulary, protected_values, sha, save

ANCHORS = {
    'items': dict(singular='item', plural='items', nodes=['n:03588414','n:03748886'], max_distance=3),
    'customers': dict(singular='customer', plural='customers', nodes=['n:09984659'], max_distance=3),
    'orders': dict(singular='order', plural='orders', nodes=['n:07279687','n:00079018'], max_distance=5),
}
COMMERCE = re.compile(r'\b(?:commercial|financial|purchase|purchases|payment|payments|buying|selling|business|goods|services|merchandise|sales)\b', re.I)
MEASURE = r'(?P<measure>(?:ordered\s+)?(?:quantity|stock(?:\s+(?:level|units?))?|price))'


def phrase_rules():
    # These are schema-level equivalences, not per-question edits. Comparators
    # max/min are deliberately excluded ("max 5" can mean a filter boundary).
    rows = [
        ('record_count',r'\brecord\s+counts?\b','row count'),
        ('count_records',r'\bnumber\s+of\s+records\b','number of rows'),
        ('arithmetic_mean',r'\barithmetic\s+mean\s+of\b','average'),
        ('mean_value',r'\bmean\s+value\s+of\b','average'),
        ('avg_of',r'\bavg\.?\s+of\b','average'),
        ('avg',r'\bavg\b','average'),
        ('qty',r'\bqty\b','quantity'),
        ('asc',r'\basc\b','ascending'),
        ('desc',r'\bdesc\b','descending'),
        ('combined_measure',r'\bcombined\s+'+MEASURE+r'\b',r'total \g<measure>'),
        ('largest_measure',r'\blargest\s+'+MEASURE+r'\b',r'highest \g<measure>'),
        ('smallest_measure',r'\bsmallest\s+'+MEASURE+r'\b',r'lowest \g<measure>'),
    ]
    return [dict(id=i,pattern=p,replacement=r) for i,p,r in rows]


def symbol_rules():
    return [dict(id='symbol_'+name,pattern=r'\s*(?<![<>=!])'+re.escape(symbol)+r'(?![<>=!])\s*',replacement=' '+text+' ')
            for name,symbol,text in [('ge','>=','at least'),('le','<=','at most'),('ne','!=','not equal to'),
                                     ('gt','>','more than'),('lt','<','less than'),('eq','=','equal to')]]


def distances(graph, starts, maximum):
    d = {s:0 for s in starts}
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        if d[node] == maximum:
            continue
        for neighbor in sorted(graph.edges[node]):
            if neighbor not in d:
                d[neighbor] = d[node]+1
                queue.append(neighbor)
    return d


def domain_mapping(graph, counts, glosses):
    paths = {t:distances(graph, a['nodes'], a['max_distance']) for t,a in ANCHORS.items()}
    mapping, audit = {}, []
    for word in graph.candidates():
        if word in counts:
            continue
        candidates = []
        for node, number in graph.senses(word):
            if not node.startswith('n:'):
                continue
            # Some plural spellings also have an independent singular synset
            # ("transactions" = meeting records). Number belongs to THIS sense.
            number = 'plural' if any(b != word and node in graph.words['n'][b]
                                     for b in graph.bases(word,'n')) else 'singular'
            for table, anchor in ANCHORS.items():
                if node not in paths[table]:
                    continue
                # A shop order is a commercial event/request, not an arbitrary
                # command, legal statement or sequential arrangement.
                if table == 'orders' and not COMMERCE.search(glosses[node]):
                    continue
                target = anchor[number]
                if target in counts:
                    candidates.append((paths[table][node],table,target,node,number))
        if not candidates:
            continue
        distance = min(c[0] for c in candidates)
        closest = [c for c in candidates if c[0] == distance]
        if len({c[1] for c in closest}) != 1:
            continue  # same-distance cross-table ambiguity: retain original
        d, table, target, node, number = min(closest)
        mapping[word] = [target,d]
        audit.append(dict(source=word,target=target,table=table,distance=d,source_synset=node,
                          gloss=glosses[node],number=number))
    return mapping,audit


def build(args):
    start = time.perf_counter()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite a frozen map')
    counts, n = train_vocabulary(args.train_files)
    graph = WordNetGraph(args.wordnet)
    with zipfile.ZipFile(args.wordnet) as z:
        glosses = {'n:'+line.split()[0]:line.split('|',1)[1].strip()
                   for line in z.read('wordnet/data.noun').decode().splitlines() if line and line[0].isdigit()}
    mapping,audit = domain_mapping(graph,counts,glosses)
    protection = protected_values(args.db)
    profiles = {'domain':(mapping,[]),'phrases':({},phrase_rules()),
                'combined':(mapping,phrase_rules()),'combined_symbols':(mapping,phrase_rules()+symbol_rules())}
    provenance = dict(training_pair_count=n, training_files={str(p):sha(p) for p in args.train_files},
        wordnet_sha256=sha(args.wordnet),db_sha256=sha(args.db),domain_anchors=ANCHORS,
        commercial_sense_filter=COMMERCE.pattern,evaluation_inputs_used=False,
        inference_uses_gold=False, protect_like_fragments=True,like_guard_version=3,
        limitations='Context-free noun sense search; human-defined domain sense anchors and phrase equivalences. Evaluations already observed: development experiment, not sealed testing.')
    out.mkdir(parents=True,exist_ok=True)
    for name,(words,rules) in profiles.items():
        save(out/f'{name}.json',dict(format_version=1,metadata=dict(provenance,variant=name),
            trained_words=sorted(counts),protected_values=protection,mapping=words,phrase_rules=rules))
    with (out/'mapping_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    manifest = dict(provenance,elapsed_s=time.perf_counter()-start,domain_words=len(mapping),
        map_bytes={k:(out/f'{k}.json').stat().st_size for k in profiles},map_sha256={k:sha(out/f'{k}.json') for k in profiles},
        profile_rules={k:len(v[1]) for k,v in profiles.items()})
    save(out/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-files',nargs='+',required=True)
    p.add_argument('--wordnet',required=True)
    p.add_argument('--db',required=True)
    p.add_argument('--out',required=True)
    build(p.parse_args())


if __name__ == '__main__': main()
