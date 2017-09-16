from __future__ import print_function

import dynet as dy
from layers import Embeddings, BiLSTM, MultiLayerPerceptron

class MSTParser(object):

    def __init__(self, model, **kwargs):
        self.pc = model.add_subcollection()

        embeddings_basename = "../build/cs"
        lstm_num_layers = 3
        lstm_dim = 100

        self.embeddings = Embeddings.init_from_word2vec(self.pc, embeddings_basename)
        input_dim = self.embeddings.dim
        self.lstm = BiLSTM(self.pc, input_dim, lstm_dim, lstm_num_layers)

        self.kwargs = kwargs
        self.spec = kwargs,

    def predict_arcs(self, tree):
        x = self.embeddings(tree)
        h = self.lstm(x)
        return h
    
    def predict_labels(self, tree):
        pass

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
 
    dy.save("../build/model", [mst])

    m2 = dy.ParameterCollection()
    mst, = dy.load("../build/model", m2)
    y = mst.predict_arcs(tree)
    print(y[0].value())
    print(len(y))
 
