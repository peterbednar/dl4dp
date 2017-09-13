from __future__ import print_function

import dynet as dy
import numpy as np
from utils import FORM, UPOS, FEATS
from word2vec import read_word2vec

class Embeddings(object):

    def __init__(self, model, dims, dropout=0):
        self.pc = model.add_subcollection()
        self.lookup = [self.pc.add_lookup_parameters(dim) for dim in dims]
        self.dropout = dropout
        self.spec = (dims, dropout)

    def __call__(self, tree):
        num_tokens, num_feats = tree.feats.shape

        def _lookup(i):
            feats = [self.lookup[f][tree.feats[i,f]] for f in range(num_feats)]
            if isinstance(self.dropout, (tuple, list)):
                feats = [dy.dropout(feats[i], p) if p > 0 else feats[i] for i, p in enumerate(self.dropout)]
                x_i = dy.concatenate(feats)
            else:
                x_i = dy.concatenate(feats)
                if self.dropout > 0:
                    x_i = dy.dropout(x_i, self.dropout)
            return x_i

        x = [_lookup(i) for i in range(num_tokens)]
        return x

    def set_dropout(self, dropout):
        self.dropout = dropout
    
    def disable_dropout():
        self.dropout = 0

    def param_collection(self):
        return self.pc

    @staticmethod
    def init_from_array(model, arrays, dropout=0):
        embeddings = Embeddings(model, [a.shape for a in arrays], dropout)
        for param, a in zip(embeddings.lookup, arrays):
            param.init_from_array(a)
        return embeddings

    @staticmethod
    def init_from_word2vec(model, basename, fields=(FORM, UPOS, FEATS), dropout=0):
        wv = read_word2vec(basename, fields)
        return Embeddings.init_from_array(model, wv, dropout)

    @staticmethod
    def from_spec(spec, model):
        dims, dropout = spec
        return Embeddings(model, dims, dropout)

from utils import DepTree

if __name__ == "__main__":
    m1 = dy.ParameterCollection()
    embeddings = Embeddings.init_from_word2vec(m1, "../build/cs")
    dy.save("../build/model", [embeddings])

    m2 = dy.ParameterCollection()
    embeddings, = dy.load("../build/model", m2)

    tree = DepTree(2, 3)
    tree.feats[:,:] = [[1,2,3],[4,5,6]]
    x = embeddings(tree)
    print(x[0].dim())
    print(x[0].value())
