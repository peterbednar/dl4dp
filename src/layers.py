from __future__ import print_function

import dynet as dy
import numpy as np
from utils import FORM, UPOS, FEATS
from word2vec import read_word2vec

class Embeddings(object):

    def __init__(self, model, dims):
        self.pc = model.add_subcollection()
        self.lookup = [self.pc.add_lookup_parameters(dim) for dim in dims]
        self.spec = tuple(dims)

    def __call__(self, tree):
        num_tokens, num_feats = tree.feats.shape
        x = []
        for i in range(num_tokens):
            x.append(dy.concatenate([self.lookup[f][tree.feats[i,f]] for f in range(num_feats)]))
        return x

    def param_collection(self):
        return self.pc

    @staticmethod
    def init_from_array(model, arrays):
        embeddings = Embeddings(model, [a.shape for a in arrays])
        for param, a in zip(embeddings.lookup, arrays):
            param.init_from_array(a)
        return embeddings

    @staticmethod
    def init_from_word2vec(model, basename, fields=(FORM, UPOS, FEATS)):
        wv = read_word2vec(basename, fields)
        return Embeddings.init_from_array(model, wv)

    @staticmethod
    def from_spec(spec, model):
        dims = spec
        return Embeddings(dims, model)

from utils import DepTree

if __name__ == "__main__":
    m = dy.ParameterCollection()
    embeddings = Embeddings.init_from_word2vec(m, "../build/cs")
    dy.save("../build/model", [embeddings])

    tree = DepTree(2, 3)
    tree.feats[:,:] = [[1,2,3],[4,5,6]]
    x = embeddings(tree)
    print(x[0].dim())
    print(x[0].value())
