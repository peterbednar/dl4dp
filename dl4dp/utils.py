import math
import heapq
import time
import re
import requests
import tarfile
from datetime import timedelta
from collections import defaultdict
from functools import total_ordering
from csv import DictWriter
from pathlib import Path
import numpy as np


_home_dir = Path.home() / '.dl4dp'


def home_dir():
    path = _home_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_home_dir(path):
    _home_dir = Path(path)


def _get_dir(dir, treebank=None, create=False):
    path = home_dir() / dir
    if treebank is not None:
        path /= treebank
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_treebank_dir(treebank=None, create=False):
    return _get_dir('treebanks', treebank, create)


def get_build_dir(treebank, create=False):
    return _get_dir('build', treebank, create)


def get_model_dir(create=False):
    return _get_dir('models', None, create)


class ConfigError(Exception):
    pass


_FILE_NAME = re.compile(r'.*-(train|test|dev).conllu')


def get_treebank_files(treebank, extract_ud=True):
    tb_dir = get_treebank_dir(treebank)

    if not tb_dir.exists() and extract_ud:
        return extract_ud_treebank(treebank)

    files = {}
    for name in tb_dir.iterdir():
        match = _FILE_NAME.match(str(name))
        if match:
            files[match.group(1)] = name

    if not files:
        raise ConfigError(f'no data found in treebank directory {tb_dir}')

    return files


_UD_FILE = 'ud-treebanks-v2.7.tgz'
_UD_URL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/' \
          'ud-treebanks-v2.7.tgz?sequence=1&isAllowed=y'


def _match_ud_treebank_name(treebank, name):
    return re.compile(fr'.*\/{treebank}-ud-(train|test|dev).conllu').match(name)


def extract_ud_treebank(treebank):
    td_dir = get_treebank_dir(create=True)
    archive = td_dir / _UD_FILE

    if not archive.exists():
        print('downloading ' + _UD_FILE)
        get_url(_UD_URL, archive)

    files = {}
    print(f'extracting {treebank} from UD archive...')

    with tarfile.open(archive, 'r', encoding='utf-8') as tar:
        extract_members = {}
        for member in tar.getmembers():
            match = _match_ud_treebank_name(treebank, member.name)
            if match:
                extract_members[match.group(1)] = member

        if not extract_members:
            raise ConfigError(f'treebank {treebank} not found in the UD archive')

        td_dir = get_treebank_dir(treebank, create=True)
        for f, member in extract_members.items():
            member.name = Path(member.name).name  # Extract only file name without path
            tar.extract(member, td_dir)
            files[f] = td_dir / member.name

    print('extracting done')
    return files


def get_url(url, path, progress=True):
    request = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        if progress:
            total_length = int(request.headers.get('content-length'))
            pb = progressbar(total_length)
        for chunk in request:
            f.write(chunk)
            if progress:
                pb.update(len(chunk))
        if progress:
            pb.finish()


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
        inverted[v-1] = int(node)
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

    def __init__(self, total=100, width=50, bar='.', end='\n'):
        self.total = total
        self.value = 0
        self.start_time = time.time()
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

    def finish(self):
        print(self.end, end='', flush=True)

    def elapsed_time(self):
        return timedelta(seconds=time.time() - self.start_time)

    def print_elapsed_time(self, unit='samples'):
        ts = time.time() - self.start_time
        td = timedelta(seconds=round(ts))
        ps = float(self.value) / ts
        print(f'elapsed time: {td}, {ps:.2f} {unit}/s', flush=True)

    def reset(self, value=0, total=None):
        self.value = value
        self.start_time = time.time()
        if total is not None:
            self.total = total


_loggers = {}


def get_logger(name):
    return _loggers[name]


def register_logger(name, filename):
    if name in _loggers:
        _loggers[name].close()
    _loggers[name] = CsvLogger(filename)


class CsvLogger(object):

    def __init__(self, filename):
        self.filename = filename
        self.writer = None
        self.stream = None

    def log(self, data):
        if self.writer is None:
            fieldnames = data.keys()
            self.stream = open(self.filename, 'w', encoding='utf8', newline='')
            self.writer = DictWriter(self.stream, fieldnames)
            self.writer.writeheader()
        self.writer.writerow(data)
        self.stream.flush()

    def close(self):
        if self.stream:
            self.stream.close()
