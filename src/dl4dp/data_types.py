from __future__ import print_function

import numpy as np
from collections import namedtuple

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

EMPTY = 0
MULTIWORD = 1

def isempty(token):
    if isinstance(token, list):
        token = token[ID]
    return token[2] == EMPTY if isinstance(token, tuple) else False

def ismultiword(token):
    if isinstance(token, list):
        token = token[ID]
    return token[2] == MULTIWORD if isinstance(token, tuple) else False

class DepTree(namedtuple("DepTree", "feats, heads, labels")):

    def __new__(cls, shape):
        return super(cls, DepTree).__new__(cls,
                np.empty(shape, dtype=np.int),
                np.full(shape[0], -1, dtype=np.int),
                np.full(shape[0], -1, dtype=np.int))

def map_to_instance(sentences, index, fields=[FORM, UPOS, FEATS]):
    f_num = len(fields)
    for sentence in sentences:
        l = len(sentence)
        tree = DepTree((l, f_num))

        for i, token in enumerate(sentence):
            for j, f in enumerate(fields):
                tree.feats[i][j] = index[f][token[f]]
                if tree.feats[i][j] == 0:
                    print(token[f])
                    
            tree.heads[i] = token[HEAD]
            tree.labels[i] = index[DEPREL][token[DEPREL]]

        yield tree
