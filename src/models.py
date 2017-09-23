from __future__ import print_function

import dynet as dy
import numpy as np
from layers import Embeddings, BiLSTM, MultiLayerPerceptron
from utils import DEPREL, read_index, max_branching, DepTree
from abc import ABCMeta, abstractmethod

class MSTParser(object):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()
        self.kwargs = kwargs

        basename = kwargs.get("basename")
        index = read_index(basename)
        self._num_labels = len(index[DEPREL])

        lstm_num_layers = kwargs.get("lstm_num_layers", 3)
        lstm_dim = kwargs.get("lstm_dim", 100)
        self.embeddings = Embeddings.init_from_word2vec(self.pc, basename, index=index)
        input_dim = self.embeddings.dim
        self.lstm = BiLSTM(self.pc, input_dim, lstm_dim, lstm_num_layers)

        self.spec = kwargs,

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
        dy.renew_cg()
        x = self.embeddings(feats)
        h = self.lstm(x)
        tree = DepTree(len(x))
        self._parse_heads(tree.heads, h)
        self._parse_labels(tree.heads, tree.labels, h)
        return tree

    def disable_dropout(self):
        self.embeddings.disable_dropout()
        self.lstm.disable_dropout()

    def enable_dropout(self):
        self.embeddings.set_dropout(self.kwargs.get("input_dropout", 0))
        self.lstm.set_dropout(self.kwargs.get("lstm_dropout", 0))

    def param_collection(self):
        return self.pc

    __metaclass__ = ABCMeta

class MLPParser(MSTParser):

    def __init__(self, model, **kwargs):
        super(MLPParser, self).__init__(model, **kwargs)
        lstm_dim = self.lstm.dims[1]
        self.arc_mlp = _build_mlp(self.pc, kwargs, "arc_mlp", lstm_dim * 2, 100, 1, 1, "relu")
        self.label_mlp = _build_mlp(self.pc, kwargs, "label_mlp", lstm_dim * 2, 100, self._num_labels, 1, "relu")

    def _predict_arc(self, head, dep, h):
        x = dy.concatenate([h[head], h[dep]])
        y = self.arc_mlp(x)
        return y

    def _predict_labels(self, head, dep, h):
        x = dy.concatenate([h[head], h[dep]])
        y = self.label_mlp(x)
        return y

    def disable_dropout(self):
        super(MLPParser, self).disable_dropout()
        self.arc_mlp.disable_dropout()
        self.label_mlp.disable_dropout()

    def enable_dropout(self):
        super(MLPParser, self).enable_dropout()
        self.arc_mlp.set_dropout(self.kwargs.get("arc_mlp_dropout", 0))
        self.label_mlp.set_dropout(self.kwargs.get("label_mlp_dropout", 0))        

    @staticmethod
    def from_spec(spec, model):
        kwargs, = spec
        return MLPParser(model, **kwargs)

_STR_TO_ACT = {"tanh": dy.tanh, "sigmoid": dy.logistic, "relu": dy.rectify}

def _build_mlp(model, kwargs, prefix, input_dim, hidden_dim, output_dim, num_layers, act):
    hidden_dim = kwargs.get(prefix + "_dim", hidden_dim)
    num_layers = kwargs.get(prefix + "_num_layers", num_layers)
    act = _STR_TO_ACT[kwargs.get(prefix + "_act", act)]
    dims = [input_dim] + [hidden_dim]*num_layers + [output_dim]
    return MultiLayerPerceptron(model, dims, act)
