from __future__ import print_function

import dynet as dy
import numpy as np
from utils import FORM, UPOS, FEATS
from word2vec import read_word2vec

class Embeddings(object):

    def __init__(self, model, dims, dropout=0, update=True):
        self.pc = model.add_subcollection()
        self.lookup = [self.pc.add_lookup_parameters(dim) for dim in dims]
        self.set_dropout(dropout)
        self.set_update(update)
        self.spec = (dims, dropout, update)

    def __call__(self, tree):

        def _lookup(i, f):
            update = self.update[f]
            dropout = self.dropout[f]
            feats = dy.lookup(self.lookup[f], tree.feats[i,f], update=update)
            if dropout > 0:
                feats = dy.dropout(feats, dropout)
            return feats

        num_tokens, num_feats = tree.feats.shape
        x = [dy.concatenate([_lookup(i, f) for f in range(num_feats)]) for i in range(num_tokens)]
        return x

    def set_update(self, update):
        self.update = update if isinstance(update, (tuple, list)) else [update] * len(self.lookup)

    def set_dropout(self, dropout):
        self.dropout = dropout if isinstance(dropout, (tuple, list)) else [dropout] * len(self.lookup)
    
    def disable_dropout():
        self.set_dropout(0)

    def param_collection(self):
        return self.pc

    @staticmethod
    def init_from_array(model, arrays, dropout=0, update=True):
        embeddings = Embeddings(model, [a.shape for a in arrays], dropout, update)
        for param, a in zip(embeddings.lookup, arrays):
            param.init_from_array(a)
        return embeddings

    @staticmethod
    def init_from_word2vec(model, basename, fields=(FORM, UPOS, FEATS), dropout=0, update=True):
        wv = read_word2vec(basename, fields)
        return Embeddings.init_from_array(model, wv, dropout, update)

    @staticmethod
    def from_spec(spec, model):
        dims, dropout, update = spec
        return Embeddings(model, dims, dropout, update)


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
