# 역할: 학습 어휘와 WordNet 의미 관계를 읽어 단어 치환 사전을 만든다.
# 사용처: 추론 전에 전처리용 JSON 사전과 보호할 DB 값 목록을 생성할 때 사용한다.

"""Build train-vocabulary WordNet maps offline; no evaluation inputs or NLTK.

Relations: identical synsets, or undirected noun/verb hypernym edges (@/@i).
Rank: shortest distance, highest training count, lexical target/synset order.
All senses are searched; there is deliberately no contextual disambiguation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import sqlite3
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from src.inference.wordnet_normalizer import WORD, QUOTED
from src.data.json_io import read_json

POS_FILES = {'n': 'noun', 'v': 'verb', 'a': 'adj', 'r': 'adv'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')


def train_vocabulary(paths):
    counts, n = Counter(), 0
    # Training labels are used ONLY offline to remove entity values from the
    # training vocabulary. Runtime normalization accepts only a question string.
    patterns = {}
    for path in paths:
        for row in read_json(path):
            text = QUOTED.sub(' ', row['question'].lower())
            literals = tuple(sorted({v.strip('%').lower() for v in re.findall(r"'([^']*)'", row['sql']) if v.strip('%')}, key=lambda v: (-len(v), v)))
            if literals:
                if literals not in patterns:
                    patterns[literals] = re.compile('|'.join(r'(?<!\w)' + re.escape(v) + r'(?!\w)' for v in literals))
                text = patterns[literals].sub(' ', text)
            counts.update(m.group().lower() for m in WORD.finditer(text))
            n += 1
    return counts, n


class WordNetGraph:
    def __init__(self, archive):
        self.words = {p: defaultdict(set) for p in POS_FILES}
        self.edges = defaultdict(set)
        self.nodes = set()
        self.exceptions = {}
        with zipfile.ZipFile(archive) as z:
            self.license = z.read('wordnet/LICENSE')
            for pos, filename in POS_FILES.items():
                for line in z.read('wordnet/data.' + filename).decode('utf-8').splitlines():
                    if not line or not line[0].isdigit():
                        continue
                    fields = line.split('|', 1)[0].split()
                    node = pos + ':' + fields[0]
                    self.nodes.add(node)
                    count = int(fields[3], 16)
                    for raw in fields[4:4 + count * 2:2]:
                        word = re.sub(r'\((?:a|p|ip)\)$', '', raw)
                        if re.fullmatch('[a-z]+', word):
                            self.words[pos][word].add(node)
                    index = 4 + count * 2
                    pointers = int(fields[index])
                    for offset in range(index + 1, index + 1 + pointers * 4, 4):
                        symbol, target_offset, target_pos, _ = fields[offset:offset+4]
                        if pos in ('n', 'v') and symbol in ('@', '@i') and target_pos == pos:
                            target = pos + ':' + target_offset
                            self.edges[node].add(target)
                            self.edges[target].add(node)
                exc = {}
                for line in z.read('wordnet/' + filename + '.exc').decode('utf-8').splitlines():
                    form, *bases = line.split()
                    exc[form] = bases
                self.exceptions[pos] = exc

    def bases(self, word, pos):
        candidates = {word, *self.exceptions[pos].get(word, [])}
        suffixes = {
            'n': [('s',''),('ses','s'),('xes','x'),('zes','z'),('ches','ch'),('shes','sh'),('men','man'),('ies','y')],
            'v': [('s',''),('ies','y'),('es','e'),('es',''),('ed','e'),('ed',''),('ing','e'),('ing','')],
            'a': [('er',''),('est',''),('er','e'),('est','e')], 'r': [],
        }
        for suffix, ending in suffixes[pos]:
            if word.endswith(suffix) and len(word) > len(suffix):
                candidates.add(word[:-len(suffix)] + ending)
        return {w for w in candidates if w in self.words[pos]}

    def senses(self, word):
        for pos in POS_FILES:
            bases = self.bases(word, pos)
            mode = ('plural' if word not in self.words['n'] and any(b != word for b in bases) else 'singular') if pos == 'n' else 'any'
            for node in sorted({s for b in bases for s in self.words[pos][b]}):
                yield node, mode

    def candidates(self):
        words = {w for d in self.words.values() for w in d}
        words.update(w for d in self.exceptions.values() for w in d if re.fullmatch('[a-z]+', w))
        # Regular noun plurals let unseen user forms such as "patrons" match
        # trained plural nouns. The same deterministic morphology handles targets.
        for w in self.words['n']:
            if w.endswith('y') and len(w) > 1 and w[-2] not in 'aeiou':
                words.add(w[:-1] + 'ies')
            elif w.endswith(('s','x','z','ch','sh')):
                words.add(w + 'es')
            else:
                words.add(w + 's')
        return sorted(words)


def nearest(graph, counts, max_distance, mode):
    best = {}
    queue = []
    for word, count in sorted(counts.items()):
        for node, word_mode in graph.senses(word):
            if node.startswith('n:') and word_mode != mode:
                continue
            if not node.startswith('n:') and mode != 'any':
                continue
            rank = (0, -count, word, node)
            if node not in best or rank < best[node]:
                best[node] = rank
                heapq.heappush(queue, (*rank, node))
    while queue:
        distance, neg_count, word, target, node = heapq.heappop(queue)
        rank = (distance, neg_count, word, target)
        if best[node] != rank or distance >= max_distance:
            continue
        for adjacent in sorted(graph.edges[node]):
            candidate = (distance + 1, neg_count, word, target)
            if adjacent not in best or candidate < best[adjacent]:
                best[adjacent] = candidate
                heapq.heappush(queue, (*candidate, adjacent))
    return best


def protected_values(db):
    result = set()
    # Runtime's real database catalog, not gold SQL or evaluation literals.
    with sqlite3.connect(f'file:{Path(db).resolve().as_posix()}?mode=ro', uri=True) as con:
        for table, columns in {'customers':['name','city','membership'], 'items':['item_name','category'], 'orders':['status']}.items():
            for column in columns:
                result.update(str(r[0]).lower() for r in con.execute(f'SELECT DISTINCT {column} FROM {table}'))
    return sorted(result)


def build(args):
    started = time.perf_counter()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a frozen map: {out}')
    counts, pair_count = train_vocabulary(args.train_files)
    graph = WordNetGraph(args.wordnet)
    print(f'Training: {pair_count} questions, {len(counts)} distinct non-literal words; WordNet {len(graph.nodes)} senses', flush=True)
    tables = {mode: nearest(graph, counts, args.max_distance, mode) for mode in ['any','singular','plural']}
    mappings = {'synonyms': {}, 'path': {}}
    audit = []
    for word in graph.candidates():
        if word in counts:
            continue
        options = []
        for node, mode in graph.senses(word):
            if node in tables[mode]:
                options.append((*tables[mode][node], node, mode))
        if not options:
            continue
        distance, neg_count, target, target_node, source_node, mode = min(options)
        mappings['path'][word] = [target, distance]
        if distance == 0:
            mappings['synonyms'][word] = [target, distance]
        audit.append(dict(source=word, target=target, distance=distance, similarity=1/(distance+1),
                          training_frequency=-neg_count, source_synset=source_node, target_synset=target_node, noun_number=mode))
    protection = protected_values(args.db)
    provenance = dict(training_pair_count=pair_count, training_files={str(p):sha(p) for p in args.train_files},
        wordnet_sha256=sha(args.wordnet), wordnet_version='3.0',
        wordnet_url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip',
        db_sha256=sha(args.db), training_vocabulary='lowercase word occurrences; quoted strings and training-SQL literals removed; static corpus, unweighted',
        similarity='max over all compatible POS synset pairs: 1/(1+undirected hypernym path length); noun number matched; no contextual disambiguation',
        rank='distance ASC, training_frequency DESC, target spelling ASC, target synset ASC',
        max_distance=args.max_distance, min_similarity=1/(args.max_distance+1), evaluation_inputs_used=False,
        protect_like_fragments=True, like_guard_version=3)
    out.mkdir(parents=True, exist_ok=True)
    for name, mapping in mappings.items():
        metadata = dict(provenance, variant=name, max_distance=0 if name=='synonyms' else args.max_distance,
                        min_similarity=1 if name=='synonyms' else 1/(args.max_distance+1), mapping_count=len(mapping))
        save(out / f'{name}.json', dict(format_version=1, metadata=metadata, trained_words=sorted(counts),
                                       protected_values=protection, mapping=mapping))
    save(out / 'training_vocabulary.json', dict(counts.most_common()))
    with (out / 'mapping_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    summary = dict(provenance, mapping_counts={k:len(v) for k,v in mappings.items()}, protected_catalog_values=len(protection),
                   map_bytes={name:(out/f'{name}.json').stat().st_size for name in mappings}, elapsed_s=time.perf_counter()-started,
                   map_sha256={name:sha(out/f'{name}.json') for name in mappings})
    save(out / 'manifest.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-files', nargs='+', required=True)
    p.add_argument('--wordnet', required=True, help='Official WordNet zip containing wordnet/data.* and *.exc')
    p.add_argument('--db', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-distance', type=int, default=2, choices=range(0,5))
    build(p.parse_args())


if __name__ == '__main__':
    main()
