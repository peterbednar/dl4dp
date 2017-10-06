from __future__ import print_function

import dynet as dy
import numpy as np
from layers import Embeddings, BiLSTM, MultiLayerPerceptron
from utils import FORM, XPOS, DEPREL, read_index, parse_nonprojective, DepTree
from abc import ABCMeta, abstractmethod

class MSTParser(object):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()
        self.kwargs = kwargs

        embeddings_dims = kwargs.get("embeddings_dims")
        self.labels_dim = kwargs.get("labels_dim")
        self.embeddings = Embeddings(self.pc, embeddings_dims)

        input_dim = self.embeddings.dim
        lstm_num_layers = kwargs.get("lstm_num_layers", 2)
        lstm_dim = kwargs.get("lstm_dim", 250)
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
        scarray = np.transpose(np.vstack([np.zeros(len(h))] + [s.npvalue() for s in scores]))
        parse_nonprojective(scarray, heads)

    def _parse_labels(self, heads, labels, h):
        scores = self.predict_labels(heads, h)
        labels[:] = [np.argmax(s.npvalue()) + 1 for s in scores]

    def parse(self, feats):
        h = self.transduce(feats)
        tree = DepTree(len(feats))
        self._parse_heads(tree.heads, h)
        self._parse_labels(tree.heads, tree.labels, h)
        dy.renew_cg()
        return tree

    def disable_dropout(self):
        self.embeddings.disable_dropout()
        self.lstm.disable_dropout()

    def enable_dropout(self):
        self.embeddings.set_dropout(self.kwargs.get("embeddings_dropout", 0), self.kwargs.get("index_dropout", None))
        self.lstm.set_dropout(self.kwargs.get("lstm_dropout", 0))

    def param_collection(self):
        return self.pc

    __metaclass__ = ABCMeta

_STR_TO_ACT = {"tanh": dy.tanh, "sigmoid": dy.logistic, "relu": dy.rectify}

def _build_mlp(model, kwargs, prefix, input_dim, hidden_dim, output_dim, num_layers, act):
    hidden_dim = kwargs.get(prefix + "_dim", hidden_dim)
    num_layers = kwargs.get(prefix + "_num_layers", num_layers)
    act = _STR_TO_ACT[kwargs.get(prefix + "_act", act)]
    return MultiLayerPerceptron(model, input_dim, hidden_dim, output_dim, num_layers, act)

class MLPParser(MSTParser):

    def __init__(self, model, **kwargs):
        super(MLPParser, self).__init__(model, **kwargs)
        lstm_dim = self.lstm.dims[1]
        self.arc_mlp = _build_mlp(self.pc, kwargs, "arc_mlp", lstm_dim * 2, 100, 1, 1, "tanh")
        self.label_mlp = _build_mlp(self.pc, kwargs, "label_mlp", lstm_dim * 2, 100, self.labels_dim, 1, "tanh")

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
