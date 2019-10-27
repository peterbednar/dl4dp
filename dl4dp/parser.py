import dynet as dy
import numpy as np

from conllutils import Instance, HEAD, DEPREL
from .layers import Embeddings, BiLSTM, Dense, Biaffine
from .utils import parse_nonprojective

_STR_TO_LOSS = {"crossentropy": dy.pickneglogsoftmax, "hinge": dy.hinge}

def _loss_and_error(scores, targets, floss):
    error = 0
    loss = []
    for i in range(len(scores)):
        pred = np.argmax(scores[i].npvalue())
        error += 1 if pred != targets[i] else 0
        loss.append(floss(scores[i], targets[i]))
    loss = dy.esum(loss)
    return loss, error

def _build_head_dep(model, kwargs, prefix, input_dim, output_dim):
    act = kwargs.get(prefix + "_act", "leaky_relu")
    dropout = kwargs.get(prefix + "_dropout", 0.33)
    head = Dense(model, input_dim, output_dim, act, dropout)
    dep = Dense(model, input_dim, output_dim, act, dropout)
    return (head, dep)

def save_model(filename, model):
    dy.save(filename, [model])

def load_model(filename, pc = None):
    if pc is None:
        pc = dy.ParameterCollection()
    model, = dy.load(filename, pc)
    return model

class BiaffineParser(object):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()
        self.kwargs = kwargs

        loss = kwargs.get("loss", "crossentropy")

        embeddings_dims = kwargs.get("embeddings_dims")
        labels_dim = kwargs.get("labels_dim")
        lstm_dim = kwargs.get("lstm_dim", 800)
        lstm_num_layers = kwargs.get("lstm_num_layers", 3)
        arc_mlp_dim = kwargs.get("arc_mlp_dim", 500)
        label_mlp_dim = kwargs.get("label_mlp_dim", 100)

        input_dropout = self.kwargs.get("input_dropout", 0.33)
        embeddings_dropout = self.kwargs.get("embeddings_dropout", 0.33)
        lstm_dropout = self.kwargs.get("lstm_dropout", 0.33) 

        self.loss = _STR_TO_LOSS[loss] if isinstance(loss, str) else loss

        self.embeddings = Embeddings(self.pc, embeddings_dims)
        self.embeddings.set_dropout(embeddings_dropout, input_dropout)

        input_dim = self.embeddings.dim
        self.lstm = BiLSTM(self.pc, input_dim, lstm_dim, lstm_num_layers, lstm_dropout, lstm_dropout, boundary_tokens=True, root_token=True)

        self.arc_head_mlp, self.arc_dep_mlp = _build_head_dep(self.pc, kwargs, "arc_mlp", lstm_dim, arc_mlp_dim)
        self.label_head_mlp, self.label_dep_mlp = _build_head_dep(self.pc, kwargs, "label_mlp", lstm_dim, label_mlp_dim)

        self.arc_biaffine = Biaffine(self.pc, arc_mlp_dim, 1, bias_y=False, bias=False)
        self.label_biaffine = Biaffine(self.pc, label_mlp_dim, labels_dim)

        self.spec = kwargs,

    def _transduce(self, example):
        x = self.embeddings(example)
        h = self.lstm(x)
        return h

    def _predict_arc(self, head, dep, h):
        x = self.arc_head_mlp(h[head])
        y = self.arc_dep_mlp(h[dep])
        return self.arc_biaffine(x, y)

    def _predict_arcs(self, h):
        num_nodes = len(h)
        def _predict_heads(dep):
            scores = [self._predict_arc(head, dep, h) if head != dep else dy.zeros(1) for head in range(num_nodes)]
            return dy.concatenate(scores)
        heads = [_predict_heads(dep) for dep in range(1, num_nodes)]
        return heads

    def _predict_label(self, head, dep, h):
        x = self.label_head_mlp(h[head])
        y = self.label_dep_mlp(h[dep])
        return self.label_biaffine(x, y)

    def _predict_labels(self, heads, h):
        num_nodes = len(h)
        labels = [self._predict_label(heads[dep-1], dep, h) for dep in range(1, num_nodes)]
        return labels

    def _arc_loss(self, example, h):
        scores = self._predict_arcs(h)
        return _loss_and_error(scores, example[HEAD], self.loss)

    def _label_loss(self, example, h):
        scores = self._predict_labels(example[HEAD], h)
        return _loss_and_error(scores, example[DEPREL] - 1, self.loss)

    def _parse_heads(self, h):
        scores = self._predict_arcs(h)
        scarray = np.transpose(np.vstack([np.zeros(len(h))] + [s.npvalue() for s in scores]))
        return parse_nonprojective(scarray)

    def _parse_labels(self, heads, h):
        scores = self._predict_labels(heads, h)
        return [np.argmax(s.npvalue()) + 1 for s in scores]

    def loss_and_errors(self, example):
        h = self._transduce(example)
        arc_loss, arc_error = self._arc_loss(example, h)
        label_loss, label_error = self._label_loss(example, h)
        return arc_loss + label_loss, np.array([arc_error, label_error], dtype=np.float)

    def parse(self, example):
        h = self._transduce(example)
        parsed = Instance(len(example), metadata=example.metadata)
        parsed[HEAD] = self._parse_heads(h)
        parsed[DEPREL] = self._parse_labels(parsed[HEAD], h)
        dy.renew_cg()
        return parsed

    def set_training(self, training):
        self.embeddings.set_training(training)
        self.lstm.set_training(training)
        for mlp in [self.arc_head_mlp, self.arc_dep_mlp, self.label_head_mlp, self.label_dep_mlp]:
            mlp.set_training(training)

    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        kwargs, = spec
        return BiaffineParser(model, **kwargs)
