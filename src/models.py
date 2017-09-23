from __future__ import print_function

import dynet as dy
import numpy as np
from layers import Embeddings, BiLSTM, MultiLayerPerceptron
from utils import DEPREL, read_index, max_branching
from abc import ABCMeta, abstractmethod

class MSTParser(object):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()

        basename = "../build/cs"
        index = read_index(basename)
        self._num_labels = len(index[DEPREL])

        lstm_num_layers = 3
        lstm_dim = 100
        self.embeddings = Embeddings.init_from_word2vec(self.pc, basename, index=index)
        input_dim = self.embeddings.dim
        self.lstm = BiLSTM(self.pc, input_dim, lstm_dim, lstm_num_layers)

    def transduce(self, feats):
        x = self.embeddings(feats)
        h = self.lstm(x)
        return h

    @abstractmethod
    def _predict_arc(self, head, dep, h):
        raise NotImplementedError()

    def predict_arcs(self, h):
        num_nodes = len(h)
        def _predict_heads(dep):
            scores = [self._predict_arc(head, dep, h) if head != dep else dy.zeros(1) for head in range(num_nodes)]
            return dy.concatenate(scores)
        heads = [_predict_heads(dep) for dep in range(1, num_nodes)]
        return heads

    @abstractmethod
    def _predict_labels(self, head, dep, h):
        raise NotImplementedError()

    def predict_labels(self, heads, h):
        num_nodes = len(h)
        labels = [self._predict_labels(heads[dep-1], dep, h) for dep in range(1, num_nodes)]
        return labels

    def _parse_heads(self, heads, h):
        scores = self.predict_arcs(h)
        weights = np.transpose(np.vstack([np.zeros(len(h))] + [s.npvalue() for s in scores]))
        max_branching(weights, heads)

    def _parse_labels(self, heads, labels, h):
        scores = self.predict_labels(heads, h)
        labels[:] = [np.argmax(scores[i].npvalue()) + 1 for i in range(len(scores))]

    def parse(self, feats):
        x = self.embeddings(feats)
        h = self.lstm(x)
        tree = DepTree(len(x))
        self._parse_heads(tree.heads, h)
        self._parse_labels(tree.heads, tree.labels, h)
        return tree

    def param_collection(self):
        return self.pc

    __metaclass__ = ABCMeta


class MLPParser(MSTParser):

    def __init__(self, model, **kwargs):
        super(MLPParser, self).__init__(model, **kwargs)
        lstm_dim = self.lstm.dims[1]

        arc_mlp_layers = 2
        arc_mlp_dim = 100
        arc_mlp_act = dy.rectify
        dims = [lstm_dim * 2] + [arc_mlp_dim]*arc_mlp_layers + [1]
        self.arc_mlp = MultiLayerPerceptron(self.pc, dims, act=arc_mlp_act)

        label_mlp_layers = 2
        label_mlp_dim = 100
        label_mlp_act = dy.rectify
        dims = [lstm_dim * 2] + [label_mlp_dim]*label_mlp_layers + [self._num_labels]
        self.label_mlp = MultiLayerPerceptron(self.pc, dims, act=label_mlp_act)

        self.kwargs = kwargs
        self.spec = kwargs,

    def _predict_arc(self, head, dep, h):
        x = dy.concatenate([h[head], h[dep]])
        y = self.arc_mlp(x)
        return y

    def _predict_labels(self, head, dep, h):
        x = dy.concatenate([h[head], h[dep]])
        y = self.label_mlp(x)
        return y

    @staticmethod
    def from_spec(spec, model):
        kwargs, = spec
        return MLPParser(model, **kwargs)

from utils import DepTree

if __name__ == "__main__":
    gold = DepTree(4, 3)
    gold.feats[:] = [[1,2,3], [4,5,6], [7,8,9], [1,2,3]]

    m1 = dy.ParameterCollection()
    mst = MLPParser(m1)

    tree = mst.parse(gold.feats)
    print(tree.heads)
    print(tree.labels)

    dy.save("../build/model", [mst])
    m2 = dy.ParameterCollection()
    mst, = dy.load("../build/model", m2)
    tree = mst.parse(gold.feats)
    print(tree.heads)
    print(tree.labels)
