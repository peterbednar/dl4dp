import heapq
import numpy as np
import sys
import math
from abc import ABC, abstractmethod
from conllutils import HEAD, DEPREL
from collections import defaultdict
from functools import total_ordering
from io import TextIOWrapper
import gzip
import lzma

@total_ordering
class Metric(ABC):

    def __init__(self):
        self.total = 0
        self.correct = 0

    @abstractmethod
    def __call__(self, gold, parsed):
        raise NotImplementedError()

    @property
    def value(self):
        return float(self.correct) / self.total

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.value:.4f}"

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

class UAS(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, gold, parsed):
        for n in range(len(gold)):
            if gold[HEAD][n] == parsed[HEAD][n]:
                self.correct += 1
            self.total += 1

class LAS(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, gold, parsed):
        for n in range(len(gold)):
            if gold[HEAD][n] == parsed[HEAD][n] and gold[DEPREL][n] == parsed[DEPREL][n]:
                self.correct += 1
            self.total += 1

class EMS(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, gold, parsed):
        self.total += 1
        for n in range(len(gold)):
            if gold[HEAD][n] != parsed[HEAD][n] or gold[DEPREL][n] != parsed[DEPREL][n]:
                return
        self.correct += 1

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

    nr, _ = scores.shape

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

def open_file(filename, encoding="utf-8", errors="strict"):
    f = open(filename, "rb")

    if filename.endswith(".gz"):
        f = gzip.open(f, "rb")
    elif filename.endswith(".xz"):
        f = lzma.open(f, "rb")

    return TextIOWrapper(f, encoding=encoding, errors=errors)

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