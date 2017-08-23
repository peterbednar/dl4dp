from __future__ import print_function

import heapq
from functools import total_ordering
from collections import defaultdict
import numpy as np

def max_branching(weights):
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
        q[node] = _EdgeQueue()
        for i in range(nr):
            if i != node:
                q[node].push(_Edge(i, node, weights[i, node]))

    while roots:
        scc_to = roots.pop()
        max_in_edge = q[scc_to].pop()

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
                q[scc_to].push(e)

            _union_disjoint_sets(s, scc_to, tmp_scc_to)
            q[tmp_scc_to] = None

            tmp = enter[_find_disjoint_sets(s, tmp.start)]

        roots.append(scc_to)

    visited = np.zeros(nr, dtype=np.bool)
    inverted = np.zeros(nr, dtype=np.int)
    for scc in rset:
        _invert_max_branching(min[scc], h, visited, inverted)

    return inverted

def _invert_max_branching(node, h, visited, inverted):
    visited[node] = True
    for v in h[node]:
        if visited[v]:
            continue
        inverted[v] = node
        _invert_max_branching(v, h, visited, inverted)

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

class _EdgeQueue(object):

    def __init__(self):
        self._queue = []

    def push(self, edge):
        heapq.heappush(self._queue, edge)

    def pop(self):
        if len(self._queue) == 0:
            return None
        return heapq.heappop(self._queue)

    def __iter__(self):
        return iter(self._queue)

    def __repr__(self):
        return str(self._queue)

def _find_disjoint_sets(trees, elm):
    if trees[elm] != elm:
        trees[elm] = _find_disjoint_sets(trees, trees[elm])
    return trees[elm]

def _union_disjoint_sets(trees, set1, set2):
    trees[set2] = set1

if __name__ == "__main__":
    weights = np.zeros((4, 4), dtype=np.float)
    weights[0, 1] = 0.5
    weights[0, 2] = 0.2
    weights[1, 3] = 0.5
    weights[0, 3] = 0.9
    heads = max_branching(weights)
    print(heads)
