import math
import heapq
from collections import defaultdict
from functools import total_ordering
import numpy as np

def tarjan(scores, heads):
    nr = scores.shape[1]

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
                _push(q[node], _edge(i, node, scores[node-1, i]))

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
    for scc in rset:
        _invert_max_branching(min[scc], h, visited, heads)

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
        inverted[v-1] = node
        _invert_max_branching(v, h, visited, inverted)

@total_ordering
class _edge(object):

    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

    def __eq__(self, other):
        return (self.weight, self.start, self.end) == (other.weight, other.start, other.end)

    def __lt__(self, other):
        return -self.weight < -other.weight

class progressbar(object):

    def __init__(self, total, width=50, bar='.', end='\n'):
        self.total = total
        self.value = 0
        self.width = width
        self.bar = bar
        self.end = end

    def _bar(self):
        return math.floor((self.value / float(self.total)) * self.width)

    def update(self, dif=1):
        prev = self._bar()
        self.value += dif
        next = self._bar()
        while prev < next:
            print(self.bar, end='', flush=True)
            prev += 1
    
    def reset(self, value=0, total=None):
        self.value = value
        if total is not None:
            self.total = total

    def finish(self):
        print(self.end, end='', flush=True)
