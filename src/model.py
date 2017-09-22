from __future__ import print_function

import dynet as dy
from layers import Embeddings, BiLSTM, MultiLayerPerceptron
from utils import DEPREL, read_index

class MSTParser(object):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()

        basename = "../build/cs"

        index = read_index(basename)
        num_labels = len(index[DEPREL])

        lstm_num_layers = 3
        lstm_dim = 100
        self.embeddings = Embeddings.init_from_word2vec(self.pc, basename, index=index)
        input_dim = self.embeddings.dim
        self.lstm = BiLSTM(self.pc, input_dim, lstm_dim, lstm_num_layers)

        arc_mlp_layers = 2
        arc_mlp_dim = 100
        arc_mlp_act = dy.rectify
        dims = [lstm_dim * 2] + [arc_mlp_dim]*arc_mlp_layers + [1]
        self.arc_mlp = MultiLayerPerceptron(self.pc, dims, act=arc_mlp_act)

        label_mlp_layers = 2
        label_mlp_dim = 100
        label_mlp_act = dy.rectify
        dims = [lstm_dim * 2] + [label_mlp_dim]*label_mlp_layers + [num_labels]
        self.label_mlp = MultiLayerPerceptron(self.pc, dims, act=label_mlp_act)

        self.kwargs = kwargs
        self.spec = kwargs,

    def predict_arcs(self, tree, h=None):
        if h is None:
            x = self.embeddings(tree)
            h = self.lstm(x)
        num_nodes = len(h)

        def _predict_arc(head, dep):
            x = dy.concatenate([h[head], h[dep]])
            y = self.arc_mlp(x)
            return y

        def _predict_heads(dep):
            scores = [_predict_arc(head, dep) if head != dep else dy.zeros(1) for head in range(num_nodes)]
            return dy.concatenate(scores)

        heads = [_predict_heads(dep) for dep in range(1, num_nodes)]
        return heads

    def predict_labels(self, tree, h=None):
        if h is None:
            x = self.embeddings(tree)
            h = self.lstm(x)
        num_nodes = len(h)

        def _predict_labels(head, dep):
            x = dy.concatenate([h[head], h[dep]])
            y = self.label_mlp(x)
            return y

        labels = [_predict_labels(tree.heads[dep-1], dep) for dep in range(1, num_nodes)]
        return labels

    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        kwargs, = spec
        return MSTParser(model, **kwargs)


from utils import DepTree

if __name__ == "__main__":

    tree = DepTree(4, 3)
    tree.feats[:] = [[1,2,3], [4,5,6], [7,8,9], [1,2,3]]

    m1 = dy.ParameterCollection()
    mst = MSTParser(m1)
    y = mst.predict_arcs(tree)
    print(y[0].value())
    print(len(y))

    y = mst.predict_labels(tree)
    print(y[0].value())
    print(len(y))
 
    dy.save("../build/model", [mst])

    m2 = dy.ParameterCollection()
    mst, = dy.load("../build/model", m2)
    y = mst.predict_arcs(tree)
    print(y[0].value())
    print(len(y))
 
    y = mst.predict_labels(tree)
    print(y[0].value())
    print(len(y))
