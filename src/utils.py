from __future__ import print_function

import codecs
import numpy as np
from collections import Counter, OrderedDict, namedtuple

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

EMPTY = 0
MULTIWORD = 1

class DepTree(namedtuple("DepTree", "feats, heads, labels")):

    def __new__(cls, shape):
        return super(cls, DepTree).__new__(cls,
                np.empty(shape, dtype=np.int),
                np.full(shape[0], -1, dtype=np.int),
                np.full(shape[0], -1, dtype=np.int))

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

def read_conllu(filename, skip_empty=True, skip_multiword=True, parse_feats=False, parse_deps=False, normalize=normalize_lower):

    def _parse_sentence(lines):
        sentence = []
        for line in lines:
            token = _parse_token(line)
            if skip_empty and isempty(token):
                continue
            if skip_multiword and ismultiword(token):
                continue
            sentence.append(token)
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
        min_fq = min_frequency[f] if isinstance(min_frequency, (list, dict)) else min_frequency
        for i, (s, fq) in enumerate(ordered):
            if fq >= min_fq:
                c[s] = i + 1
            else:
                del c[s]
    return dic

def create_inverse_index(index):
    return {f: {v: k for k, v in c.items()} for f, c in index.items()}

def map_to_instance(sentence, index, fields=[FORM, UPOS, FEATS]):
    l = len(sentence)
    f_num = len(fields)
    tree = DepTree((l, f_num))

    for i, token in enumerate(sentence):
        for j, f in enumerate(fields):
            tree.feats[i][j] = index[f][token[f]]
        tree.heads[i] = token[HEAD]
        tree.labels[i] = index[DEPREL][token[DEPREL]]

    return tree

def map_to_instances(sentences, index, fields=[FORM, UPOS, FEATS]):
    for sentence in sentences:
        yield map_to_instance(sentence, index, fields)

if __name__ == "__main__":
    dic = create_dictionary(read_conllu("../../test/test1.conllu"), fields={FORM, UPOS, FEATS, DEPREL})
    index = create_index(dic)
    print(index)
    for tree in map_to_instances(read_conllu("../../test/test1.conllu"), index):
        print(tree)
