import dynet as dy
import numpy as np
from abc import ABC, abstractmethod

from conllutils import Instance, HEAD, DEPREL
from .layers import Embeddings, BiLSTM, MultiLayerPerceptron, Dense, Biaffine
from .utils import parse_nonprojective

_STR_TO_LOSS = {"crossentropy": dy.pickneglogsoftmax, "hinge": dy.hinge}

def _error_and_loss(scores, targets, floss):
    error = 0
    loss = []
    for i in range(len(scores)):
        pred = np.argmax(scores[i].npvalue())
        error += 1 if pred != targets[i] else 0
        loss.append(floss(scores[i], targets[i]))
    loss = dy.esum(loss)
    return loss, error

class MSTParser(ABC):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()
        self.kwargs = kwargs

        self.loss = kwargs.get("loss", "crossentropy")
        if isinstance(self.loss, str):
            self.loss = _STR_TO_LOSS[self.loss]

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
    def _predict_label(self, head, dep, h):
        raise NotImplementedError()

    def predict_labels(self, heads, h):
        num_nodes = len(h)
        labels = [self._predict_label(heads[dep-1], dep, h) for dep in range(1, num_nodes)]
        return labels

    def arc_loss(self, feats, h):
        scores = self.predict_arcs(h)
        return _error_and_loss(scores, feats[HEAD], self.loss)

    def label_loss(self, feats, h):
        scores = self.predict_labels(feats[HEAD], h)
        return _error_and_loss(scores, feats[DEPREL] - 1, self.loss)

    def _parse_heads(self, h):
        scores = self.predict_arcs(h)
        scarray = np.transpose(np.vstack([np.zeros(len(h))] + [s.npvalue() for s in scores]))
        return parse_nonprojective(scarray)

    def _parse_labels(self, heads, h):
        scores = self.predict_labels(heads, h)
        return [np.argmax(s.npvalue()) + 1 for s in scores]

    def parse(self, feats):
        h = self.transduce(feats)
        tree = Instance(len(feats), metadata=feats.metadata)
        tree[HEAD] = self._parse_heads(h)
        tree[DEPREL] = self._parse_labels(tree[HEAD], h)
        dy.renew_cg()
        return tree        

    def disable_dropout(self):
        self.embeddings.disable_dropout()
        self.lstm.disable_dropout()

    def enable_dropout(self):
        self.embeddings.set_dropout(self.kwargs.get("embeddings_dropout", 0), self.kwargs.get("input_dropout", 0))
        self.lstm.set_dropout(self.kwargs.get("lstm_dropout", 0))

    def param_collection(self):
        return self.pc

    def save(self, filename):
        dy.save(filename, [self])

    @staticmethod
    def load(filename):
        pc = dy.ParameterCollection()
        model, = dy.load(filename, pc)
        return model

def _build_mlp(model, kwargs, prefix, input_dim, hidden_dim, output_dim, num_layers, act):
    hidden_dim = kwargs.get(prefix + "_dim", hidden_dim)
    num_layers = kwargs.get(prefix + "_num_layers", num_layers)
    act = kwargs.get(prefix + "_act", act)
    return MultiLayerPerceptron(model, input_dim, hidden_dim, output_dim, num_layers, act)

class MLPParser(MSTParser):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        lstm_dim = self.lstm.dims[1]
        self.arc_mlp = _build_mlp(self.pc, kwargs, "arc_mlp", lstm_dim * 2, 100, 1, 1, "tanh")
        self.label_mlp = _build_mlp(self.pc, kwargs, "label_mlp", lstm_dim * 2, 100, self.labels_dim, 1, "tanh")

    def _predict_arc(self, head, dep, h):
        x = dy.concatenate([h[head], h[dep]])
        y = self.arc_mlp(x)
        return y

    def _predict_label(self, head, dep, h):
        x = dy.concatenate([h[head], h[dep]])
        y = self.label_mlp(x)
        return y

    def disable_dropout(self):
        super().disable_dropout()
        self.arc_mlp.disable_dropout()
        self.label_mlp.disable_dropout()

    def enable_dropout(self):
        super().enable_dropout()
        self.arc_mlp.set_dropout(self.kwargs.get("arc_mlp_dropout", 0))
        self.label_mlp.set_dropout(self.kwargs.get("label_mlp_dropout", 0))        

    @staticmethod
    def from_spec(spec, model):
        kwargs, = spec
        return MLPParser(model, **kwargs)

def _build_head_dep(model, kwargs, prefix, input_dim, output_dim, act):
    act = kwargs.get(prefix + "_act", act)
    head = Dense(model, input_dim, output_dim, act)
    dep = Dense(model, input_dim, output_dim, act)
    return (head, dep)

class BiaffineParser(MSTParser):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        lstm_dim = self.lstm.dims[1]

        arc_dim = kwargs.get("arc_mlp_dim", 100)
        label_dim = kwargs.get("label_mlp_dim", 100)

        self.arc_head_mlp, self.arc_dep_mlp = _build_head_dep(self.pc, kwargs, "arc_mlp", lstm_dim, arc_dim, "leaky_relu")
        self.label_head_mlp, self.label_dep_mlp = _build_head_dep(self.pc, kwargs, "label_mlp", lstm_dim, label_dim, "leaky_relu")

        self.arc_biaffine = Biaffine(self.pc, arc_dim, 1)
        self.label_biaffine = Biaffine(self.pc, label_dim, self.labels_dim)

    def _predict_arc(self, head, dep, h):
        x = self.arc_head_mlp(h[head])
        y = self.arc_dep_mlp(h[dep])
        return self.arc_biaffine(x, y)

    def _predict_label(self, head, dep, h):
        x = self.label_head_mlp(h[head])
        y = self.label_dep_mlp(h[dep])
        return self.label_biaffine(x, y)

    def disable_dropout(self):
        super().disable_dropout()
        self.arc_head_mlp.disable_dropout()
        self.arc_dep_mlp.disable_dropout()
        self.label_head_mlp.disable_dropout()
        self.label_dep_mlp.disable_dropout()

    def enable_dropout(self):
        super().enable_dropout()
        self.arc_head_mlp.set_dropout(self.kwargs.get("arc_mlp_dropout", 0))
        self.arc_dep_mlp.set_dropout(self.kwargs.get("arc_mlp_dropout", 0))
        self.label_head_mlp.set_dropout(self.kwargs.get("label_mlp_dropout", 0))
        self.label_dep_mlp.set_dropout(self.kwargs.get("label_mlp_dropout", 0))

    @staticmethod
    def from_spec(spec, model):
        kwargs, = spec
        return BiaffineParser(model, **kwargs)
