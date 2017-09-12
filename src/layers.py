from __future__ import print_function

import dynet as dy
import numpy as np
from utils import FORM, UPOS, FEATS

class Embeddings(object):

    def __init__(self, dims, model):
        self.pc = model.add_subcollection()
        self.lookup = [self.pc.add_lookup_parameters(dim) for dim in dims]
        self.spec = tuple(dims)

    def __call__(self, tree):
        pass

    def param_collection(self):
        return self.pc

    def init_from_array(self, embeddings):
        for param, emb in zip(self.lookup, embeddings):
            param.init_from_array(emb)
        return self

    @staticmethod
    def init_from_word2vec(basename, model, fields=(FORM, UPOS, FEATS)):
        wv = read_word2vec(basename, fields)
        embeddings = Embeddings([v.shape for v in wv], model).init_from_array(wv)
        return embeddings

    @staticmethod
    def from_spec(spec, model):
        dims = spec
        return Embeddings(dims, model)

if __name__ == "__main__":
    m = dy.ParameterCollection()
    vectors = [np.random.rand(5, i * 5) for i in range(1, 5)]

    embeddings = Embeddings([v.shape for v in vectors], m)
    embeddings.init_from_array(vectors)

    dy.save("../build/model", [embeddings])
    
