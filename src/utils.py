from __future__ import print_function

import codecs
import heapq
import numpy as np
from collections import Counter, OrderedDict, namedtuple, defaultdict
from functools import total_ordering

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

EMPTY = 0
MULTIWORD = 1

FIELD_TO_STR = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
STR_TO_FIELD = {k : v for v, k in enumerate(FIELD_TO_STR)}

def _init_placeholder(id, label, head=-1):
    node = [label] * 10
    node[ID] = id
    node[HEAD] = head
    return node

ROOT_NODE = _init_placeholder(0, "__root__")
BOS_NODE = _init_placeholder(-1, "__bos__")
EOS_NODE = _init_placeholder(-1, "__eos__")

def isempty(token):
    if isinstance(token, list):
        token = token[ID]
    return token[2] == EMPTY if isinstance(token, tuple) else False

def ismultiword(token):
    if isinstance(token, list):
        token = token[ID]
    return token[2] == MULTIWORD if isinstance(token, tuple) else False
    
def normalize_lower(field, value):
    return value.lower() if field == FORM else value

def read_conllu(filename, skip_empty=True, skip_multiword=True, parse_feats=False, parse_deps=False, normalize=normalize_lower,
        insert_boundaries=True, insert_root=True):

    def _parse_sentence(lines):
        sentence = []
        if insert_boundaries:
            sentence.append(BOS_NODE)

        if insert_root:
            sentence.append(ROOT_NODE)
        for line in lines:
            token = _parse_token(line)
            if skip_empty and isempty(token):
                continue
            if skip_multiword and ismultiword(token):
                continue
            sentence.append(token)

        if insert_boundaries:
            sentence.append(EOS_NODE)
        return sentence

    def _parse_token(line):
        fields = line.split("\t")

        if "." in fields[ID]:
            token_id, index = fields[ID].split(".")
            id = (int(token_id), int(index), EMPTY)
        elif "-" in fields[ID]:
            start, end = fields[ID].split("-")
            id = (int(start), int(end), MULTIWORD)
        else:
            id = int(fields[ID])
        fields[ID] = id

        for f in [LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]:
            if fields[f] == "_":
                fields[f] = None

        if parse_feats and fields[FEATS]:
            fields[FEATS] = _parse_feats(fields[FEATS])

        if fields[HEAD]:
            fields[HEAD] = int(fields[HEAD])

        if parse_deps and fields[DEPS]:
            fields[DEPS] = _parse_deps(fields[DEPS])

        if normalize:
            for f in [FORM, LEMMA]:
                fields[f] = normalize(f, fields[f])
        
        return fields

    def _parse_feats(str):
        feats = OrderedDict()
        for key, value in [feat.split("=") for feat in str.split("|")]:
            if "," in value:
                value = value.split(",")
            feats[key] = value
        return feats

    def _parse_deps(str):
        return list(map(lambda rel: (int(rel[0]), rel[1]), [rel.split(":") for rel in str.split("|")]))

    lines = []
    with codecs.open(filename, "r", "utf-8") as fp:
        for line in fp:
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                continue
            if not line:
                if len(lines) != 0:
                    yield _parse_sentence(lines)
                    lines = []
                continue
            lines.append(line)
        if len(lines) != 0:
            yield _parse_sentence(lines)

def create_dictionary(sentences, fields={FORM, LEMMA, UPOS, XPOS, FEATS, DEPREL}):
    dic = {f: Counter() for f in fields}
    for sentence in sentences:
        for token in sentence:
            for f in fields:
                s = token[f]
                dic[f][s] += 1
    return dic

def create_index(dic, min_frequency=1):
    for f, c in dic.items():
        ordered = c.most_common()
        min_fq = min_frequency[f] if isinstance(min_frequency, (list, tuple, dict)) else min_frequency
        for i, (s, fq) in enumerate(ordered):
            if fq >= min_fq:
                c[s] = i + 1
            else:
                del c[s]
    return dic

def create_inverse_index(index):
    return {f: {v: k for k, v in c.items()} for f, c in index.items()}

INDEX_FILENAME = "{0}_{1}_index.txt"

_NONE_TOKEN = u"__none__"

def write_index(basename, index, fields={FORM, UPOS, FEATS, DEPREL}):
    index = create_inverse_index(index)
    for f in fields:
        c = index[f]
        with codecs.open(INDEX_FILENAME.format(basename, FIELD_TO_STR[f]), "w", "utf-8") as fp:
            for i in range(1, len(c) + 1):
                token = c[i]
                if token is None:
                    token = _NONE_TOKEN
                print(token, file=fp)

def read_index(basename, fields={FORM, UPOS, FEATS, DEPREL}):
    index = {}
    for f in fields:
        index[f] = Counter()
        with codecs.open(INDEX_FILENAME.format(basename, FIELD_TO_STR[f]), "r", "utf-8") as fp:
            i = 1
            for line in fp:
                token = line.rstrip("\r\n")
                if token == _NONE_TOKEN:
                    token = None
                index[f][token] = i
                i += 1
    return index

class DepTree(namedtuple("DepTree", "feats, heads, labels")):

    def __new__(cls, num_tokens, num_feats=0):
        return super(cls, DepTree).__new__(cls,
                np.empty((num_tokens, num_feats), dtype=np.int) if num_feats > 0 else None,
                np.full(num_tokens, -1, dtype=np.int),
                np.full(num_tokens, -1, dtype=np.int))

def map_to_instance(sentence, index, fields=(FORM, UPOS, FEATS)):
    num_tokens = len(sentence)
    num_feats = len(fields)
    tree = DepTree(num_tokens, num_feats)

    for i, token in enumerate(sentence):
        for j, f in enumerate(fields):
            tree.feats[i][j] = index[f][token[f]]
        tree.heads[i] = token[HEAD]
        tree.labels[i] = index[DEPREL][token[DEPREL]]

    return tree

def map_to_instances(sentences, index, fields=(FORM, UPOS, FEATS)):
    for sentence in sentences:
        yield map_to_instance(sentence, index, fields)

def max_branching(weights, branching=None):

    def _push(queue, elm):
        heapq.heappush(queue, elm)
    
    def _pop(queue):
        if len(queue) == 0:
            return None
        return heapq.heappop(queue)

    def _find_disjoint_sets(trees, elm):
        if trees[elm] != elm:
            trees[elm] = _find_disjoint_sets(trees, trees[elm])
        return trees[elm]

    def _union_disjoint_sets(trees, set1, set2):
        trees[set2] = set1

    def _invert_max_branching(node, h, visited, inverted):
        visited[node] = True
        for v in h[node]:
            if visited[v]:
                continue
            inverted[v - 1] = node
            _invert_max_branching(v, h, visited, inverted)

    nr, nc = weights.shape

    roots = list(range(1, nr))
    rset = [0]

    q = np.empty(nr, dtype=np.object)
    enter = np.empty(nr, dtype=np.object)

    min = np.arange(nr, dtype=np.int)
    s = np.arange(nr, dtype=np.int)
    w = np.arange(nr, dtype=np.int)

    h = defaultdict(list)

    for node in range(1, nr):
        q[node] = []
        for i in range(nr):
            if i != node:
                _push(q[node], _Edge(i, node, weights[i, node]))

    while roots:
        scc_to = roots.pop()
        max_in_edge = _pop(q[scc_to])

        if max_in_edge is None:
            rset.append(scc_to)
            continue

        scc_from = _find_disjoint_sets(s, max_in_edge.start)
        if scc_from == scc_to:
            roots.append(scc_to)
            continue

        h[max_in_edge.start].append(max_in_edge.end)

        wss_from = _find_disjoint_sets(w, max_in_edge.start)
        wss_to = _find_disjoint_sets(w, max_in_edge.end)
        if wss_from != wss_to:
            _union_disjoint_sets(w, wss_from, wss_to)
            enter[scc_to] = max_in_edge
            continue

        min_weight = np.inf
        min_scc = -1
        tmp = max_in_edge
        while tmp is not None:
            if tmp.weight < min_weight:
                min_weight = tmp.weight
                min_scc = _find_disjoint_sets(s, tmp.end)
            tmp = enter[_find_disjoint_sets(s, tmp.start)]

        inc = min_weight - max_in_edge.weight
        for e in q[scc_to]:
            e.weight += inc

        min[scc_to] = min[min_scc]

        tmp = enter[scc_from]
        while tmp is not None:
            inc = min_weight - tmp.weight
            tmp_scc_to = _find_disjoint_sets(s, tmp.end)
            for e in q[tmp_scc_to]:
                e.weight += inc
                _push(q[scc_to], e)

            _union_disjoint_sets(s, scc_to, tmp_scc_to)
            q[tmp_scc_to] = None

            tmp = enter[_find_disjoint_sets(s, tmp.start)]

        roots.append(scc_to)

    visited = np.zeros(nr, dtype=np.bool)
    if branching is None:
        branching = np.zeros(nr - 1, dtype=np.int)
    for scc in rset:
        _invert_max_branching(min[scc], h, visited, branching)

    return branching

@total_ordering
class _Edge(object):

    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

    def __eq__(self, other):
        return (self.weight, self.start, self.end) == (other.weight, other.start, other.end)

    def __lt__(self, other):
        return (-self.weight, self.start, self.end) < (-other.weight, other.start, other.end)

    def __repr__(self):
        return str((self.start, self.end, self.weight))
