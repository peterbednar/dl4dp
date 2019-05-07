import heapq
import numpy as np
import os
import re
import random
import sys
import math
from collections import Counter, OrderedDict, namedtuple, defaultdict
from functools import total_ordering
from io import TextIOWrapper
import gzip
import lzma

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC, FORM_NORM, LEMMA_NORM, UPOS_FEATS, \
FORM_CHARS, LEMMA_CHARS, FORM_NORM_CHARS, LEMMA_NORM_CHARS = range(17)

EMPTY = 0
MULTIWORD = 1

FIELD_TO_STR = ["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc", "form_norm", "lemma_norm", "upos_feats",
                "form_chars", "lemma_chars", "form_norm_chars", "lemma_norm_chars"]
STR_TO_FIELD = {k : v for v, k in enumerate(FIELD_TO_STR)}

def str_to_field(s):
    return STR_TO_FIELD[s.lower()]

def field_to_str(f):
    return FIELD_TO_STR[f]

def is_empty(token):
    id = token[ID]
    return id[2] == EMPTY if isinstance(id, tuple) else False

def is_multiword(token):
    id = token[ID]
    return id[2] == MULTIWORD if isinstance(id, tuple) else False
    
_NUM_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
NUM_FORM = u"__number__"
_NUM_FORM_CHARS = (u"0",)

def normalize_lower(field, value):
    if value is None:
        return None
    if field == FORM:
        return value.lower()
    return value

def normalize_default(field, value):
    if value is None:
        return None
    if field == FORM:
        if _NUM_REGEX.match(value):
            return NUM_FORM
        return value.lower()
    return value

def splitter_form(field, value):
    if value is None:
        return None
    if field == FORM:
        return tuple(value)
    return None

def splitter_default(field, value):
    if value is None:
        return None
    if field == FORM_NORM and value == NUM_FORM:
        return _NUM_FORM_CHARS
    return tuple(value)

_CHARS_FIELDS = {FORM: FORM_CHARS, LEMMA: LEMMA_CHARS, FORM_NORM: FORM_NORM_CHARS, LEMMA_NORM: LEMMA_NORM_CHARS}

def read_conllu(file, skip_empty=True, skip_multiword=True, parse_feats=False, parse_deps=False, upos_feats=True,
                normalize=normalize_default, splitter=None):

    def _parse_sentence(lines):
        sentence = []
        for line in lines:
            token = _parse_token(line)
            if skip_empty and is_empty(token):
                continue
            if skip_multiword and is_multiword(token):
                continue
            sentence.append(token)
        return sentence

    def _parse_token(line):
        fields = line.split("\t")
        fields = fields[:MISC + 1]
        fields += [None] * (LEMMA_NORM_CHARS - MISC)

        if "." in fields[ID]:
            token_id, index = fields[ID].split(".")
            id = (int(token_id), int(index), EMPTY)
        elif "-" in fields[ID]:
            start, end = fields[ID].split("-")
            id = (int(start), int(end), MULTIWORD)
        else:
            id = int(fields[ID])
        fields[ID] = id

        for f in [FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]:
            if fields[f] == "_":
                fields[f] = None

        if upos_feats:
            upos = fields[UPOS]
            feats = fields[FEATS]
            if upos:
                tag = "POS={0}|{1}".format(upos, feats) if feats else "POS={0}".format(upos)
            else:
                tag = feats
            fields[UPOS_FEATS] = tag

        if parse_feats and fields[FEATS]:
            fields[FEATS] = _parse_feats(fields[FEATS])

        if fields[HEAD]:
            fields[HEAD] = int(fields[HEAD])

        if parse_deps and fields[DEPS]:
            fields[DEPS] = _parse_deps(fields[DEPS])

        if normalize:
            fields[FORM_NORM] = normalize(FORM, fields[FORM])
            fields[LEMMA_NORM] = normalize(LEMMA, fields[LEMMA])

        if splitter:
            for (f, ch) in _CHARS_FIELDS.items():
                fields[ch] = splitter(f, fields[f])

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
    if isinstance(file, str):
        file = _open_file(file)
    with file:
        for line in file:
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
                if isinstance(s, (list, tuple)):
                    for ch in s:
                        dic[f][ch] += 1
                else:
                    dic[f][s] += 1
    return dic

def create_index(dic, min_frequency=1):
    index = {f: Counter() for f in dic.keys()}
    for f, c in dic.items():
        ordered = c.most_common()
        min_fq = min_frequency[f] if isinstance(min_frequency, (list, tuple, dict)) else min_frequency
        for i, (s, fq) in enumerate(ordered):
            if fq >= min_fq:
                index[f][s] = i + 1
    return index

def create_inverse_index(index):
    return {f: {v: k for k, v in c.items()} for f, c in index.items()}

INDEX_FILENAME = "{0}index_{1}.txt"

_NONE_TOKEN = u"__none__"

def write_index(index, fields=None, basename=""):
    if fields is None:
        fields = index.keys()
    index = create_inverse_index(index)
    for f in fields:
        filename = INDEX_FILENAME.format(basename, field_to_str(f))
        with open(filename, "wt", encoding="utf-8") as fp:
            c = index[f]
            for i in range(1, len(c) + 1):
                token = c[i]
                if token is None:
                    token = _NONE_TOKEN
                print(token, file=fp)

def read_index(fields=None, basename=""):
    if fields is None:
        fields = range(len(FIELD_TO_STR))
    index = {}
    for f in fields:
        filename = INDEX_FILENAME.format(basename, field_to_str(f))
        if os.path.isfile(filename):
            with open(filename, "rt", encoding="utf-8") as fp:
                index[f] = Counter()
                i = 1
                for line in fp:
                    token = line.rstrip("\r\n")
                    if token == _NONE_TOKEN:
                        token = None
                    index[f][token] = i
                    i += 1
    return index

def count_frequency(sentences, index, fields=None):
    if fields is None:
        fields = index.keys()
    count = {f: Counter() for f in fields}
    for sentence in sentences:
        for token in sentence:
            for f in fields:
                s = token[f]
                if isinstance(s, (list, tuple)):
                    for ch in s:
                        i = index[f][ch]
                        count[f][i] += 1
                else:
                    i = index[f][s]
                    count[f][i] += 1
    return count

def shuffled_stream(data):
    while True:
        random.shuffle(data)
        for d in data:
            yield d

class DepTree(namedtuple("DepTree", "chars, feats, heads, labels")):

    def __new__(cls, num_tokens, num_feats=0, chars_field=False):
        return super(cls, DepTree).__new__(cls,
                np.empty(num_tokens, dtype=np.object) if chars_field else None,
                np.empty((num_tokens, num_feats), dtype=np.int) if num_feats > 0 else None,
                np.full(num_tokens, -1, dtype=np.int),
                np.full(num_tokens, -1, dtype=np.int))

    def __len__(self):
        return len(self.heads)

def map_to_instance(sentence, index, fields=(FORM, UPOS, FEATS), chars_field=None):
    num_tokens = len(sentence)
    num_feats = len(fields)
    tree = DepTree(num_tokens, num_feats, chars_field is not None)

    for i, token in enumerate(sentence):
        if chars_field:
            chars = [index[chars_field][ch] for ch in token[chars_field]]
            tree.chars[i] = np.array(chars, dtype=np.int)
        for j, f in enumerate(fields):
            tree.feats[i][j] = index[f][token[f]]
        tree.heads[i] = token[HEAD]
        tree.labels[i] = index[DEPREL][token[DEPREL]]
            
    return tree

def map_to_instances(sentences, index, fields=(FORM, UPOS, FEATS), chars_field=None):
    for sentence in sentences:
        yield map_to_instance(sentence, index, fields, chars_field)

def is_projective(heads):
    n_len = heads.shape[0]
    for i in range(n_len):
        if heads[i] < 0:
            continue
        for j in range(i + 1, n_len):
            if heads[j] < 0:
                continue
            edge1_0 = min(i, heads[i])
            edge1_1 = max(i, heads[i])
            edge2_0 = min(j, heads[j])
            edge2_1 = max(j, heads[j])
            if edge1_0 == edge2_0:
                if edge1_1 == edge2_1:
                    return False
                else:
                    continue
            if edge1_0 < edge2_0 and not (edge2_0 >= edge1_1 or edge2_1 <= edge1_1):
                return False
            if edge1_0 > edge2_0 and not (edge1_0 >= edge2_1 or edge1_1 <= edge2_1):
                return False
    return True

@total_ordering
class _edge(object):

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

def parse_nonprojective(scores, heads=None):

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

    nr, nc = scores.shape

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
                _push(q[node], _edge(i, node, scores[i, node]))

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
    if heads is None:
        heads = -np.ones(nr - 1, dtype=np.int)
    for scc in rset:
        _invert_max_branching(min[scc], h, visited, heads)

    return heads

def _open_file(filename, encoding="utf-8", errors="strict"):
    f = open(filename, "rb")

    if filename.endswith(".gz"):
        f = gzip.open(f, "rb")
    elif filename.endswith(".xz"):
        f = lzma.open(f, "rb")

    return TextIOWrapper(f, encoding=encoding, errors=errors)

def open_treebank(treebank, basename=""):
    filename = basename + treebank
    return _open_file(filename)

def open_embeddings(embeddings, basename=""):
    filename = basename + embeddings
    return _open_file(filename, errors="replace")

def read_embeddings(file, skip_shape=True):

    def _tokenize(l):
        return l.rstrip(" \r\n").split(" ")

    with file:
        l = file.readline()
        if not skip_shape:
            tokens = _tokenize(l)
            yield (int(tokens[0]), int(tokens[1]))
        for l in file:
            tokens = _tokenize(l)
            w = tokens[0]
            if w == _NONE_TOKEN:
                w = None
            v = np.array([float(t) for t in tokens[1:]], dtype=np.float)
            yield (w, v) 

class progressbar(object):

    def __init__(self, total, width=50, bar=".", end="\n"):
        self.total = total
        self.value = 0
        self.width = width
        self.bar = bar
        self.end = end

    def _bar(self):
        return math.floor((self.value / float(self.total)) * self.width)

    def update(self, dif):
        prev = self._bar()
        self.value += dif
        next = self._bar()
        if prev < next:
            print(self.bar, end="")
            sys.stdout.flush()
    
    def reset(self, value=0, total=None):
        self.value = value
        if total is not None:
            self.total = total

    def finish(self):
        print(self.end, end="")
        sys.stdout.flush()

if __name__ == "__main__":
    vectors = read_embeddings(open_embeddings("vectors_form_norm.txt", "../build/"), skip_shape=False)
    print(vectors.__next__())
    for (w, v) in vectors:
        pass