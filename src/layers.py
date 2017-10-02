from __future__ import print_function

import math
import dynet as dy
from utils import FORM, UPOS, FEATS
from word2vec import read_word2vec, read_word2vec_arrays

class Embeddings(object):

    def __init__(self, model, dims, dropout=0, update=True):
        self.pc = model.add_subcollection()
        self.lookup = [self.pc.add_lookup_parameters(dim) for dim in dims]
        self.set_dropout(dropout)
        self.set_update(update)
        self.dim = sum([dim for (_,dim) in dims])
        self.spec = (dims, dropout, update)

    def __call__(self, feats):

        def _lookup(i, f):
            embds = dy.lookup(self.lookup[f], feats[i,f], update=self.update[f])
            dropout = self.dropout[f]
            if dropout > 0:
                embds = dy.dropout(embds, dropout)
            return embds

        num_tokens, num_feats = feats.shape
        if num_feats > 1:
            x = [dy.concatenate([_lookup(i,f) for f in range(num_feats)]) for i in range(num_tokens)]
        else:
            x = [_lookup(i,0) for i in range(num_tokens)]
        return x

    def set_dropout(self, dropout):
        self.dropout = dropout if isinstance(dropout, (tuple, list)) else [dropout] * len(self.lookup)
    
    def disable_dropout(self):
        self.set_dropout(0)

    def set_update(self, update):
        self.update = update if isinstance(update, (tuple, list)) else [update] * len(self.lookup)

    def param_collection(self):
        return self.pc

    @staticmethod
    def init_from_array(model, arrays, dropout=0, update=True):
        embeddings = Embeddings(model, [a.shape for a in arrays], dropout, update)
        for param, a in zip(embeddings.lookup, arrays):
            param.init_from_array(a)
        return embeddings

    @staticmethod
    def init_from_word2vec(model, basename, dims, fields=(FORM, UPOS, FEATS), index=None, dropout=0, update=True):
        embeddings = Embeddings(model, dims, dropout, update)
        for fi,i,vec in read_word2vec(basename, fields):
            embeddings.lookup[fi].init_row(i, vec)

    @staticmethod
    def from_spec(spec, model):
        dims, dropout, update = spec
        return Embeddings(model, dims, dropout, update)

class Dense(object):

    def __init__(self, model, input_dim, output_dim, act=dy.rectify, init_gain=math.sqrt(2.), ln=False):
        self.pc = model.add_subcollection()
        self.act = act
        self.ln = ln
        self.W = self.pc.add_parameters((output_dim, input_dim), init=dy.GlorotInitializer(gain=init_gain))
        self.b = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(0.))
        if ln:
            self.g = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(1.))
        self.spec = (input_dim, output_dim, act, init_gain, ln)

    def __call__(self, x):
        W = dy.parameter(self.W)
        b = dy.parameter(self.b)
        if self.ln:
            g = dy.parameter(self.g)
            y = dy.layer_norm(W * x, g, b)
            return self.act(y)
        else:
            y = dy.affine_transform([b, W, x])
            return self.act(y)
    
    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        input_dim, output_dim, act, init_gain, ln = spec
        return Dense(model, input_dim, output_dim, act, init_gain, ln)

class Identity(object):

    def __init__(self, model, input_dim, output_dim, init_gain=math.sqrt(2.)):
        self.pc = model.add_subcollection()
        self.W = self.pc.add_parameters((output_dim, input_dim), init=dy.GlorotInitializer(gain=init_gain))
        self.spec = (input_dim, output_dim, init_gain)
    
    def __call__(self, x):
        return dy.parameter(self.W) * x

    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        input_dim, output_dim, init_gain = spec
        return Dense(model, input_dim, output_dim, init_gain)

class MultiLayerPerceptron(object):

    def __init__(self, model, dims, act=dy.rectify, init_gain=math.sqrt(2.), ln=False, dropout=0):
        self.pc = model.add_subcollection()
        self.dims = tuple(dims)
        self.layers = []
        for input_dim, output_dim in zip(dims, dims[1:-1]):
            self.layers.append(Dense(self.pc, input_dim, output_dim, act, init_gain, ln))
        self.layers.append(Identity(self.pc, dims[-2], dims[-1], init_gain))
        self.dropout = dropout
        self.spec = (dims, act, init_gain, ln, dropout)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.dropout > 0:
                x = dy.dropout(x, self.dropout)
        return x

    def set_dropout(self, dropout):
        self.dropout = dropout
    
    def disable_dropout(self):
        self.set_dropout(0)

    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        dims, act, init_gain, ln, dropout = spec
        return MultiLayerPerceptron(model, dims, act, init_gain, ln, dropout)

class BiLSTM(object):

    def __init__(self, model, input_dim, hidden_dim, num_layers=1, input_dropout=0, output_dropout=0, ln=False):
        self.pc = model.add_subcollection()
        self.dims = (input_dim, hidden_dim)

        def _build_layer(input_dim, hidden_dim, rnn_builder=dy.VanillaLSTMBuilder):
            f = rnn_builder(1, input_dim, hidden_dim / 2, self.pc, ln)
            b = rnn_builder(1, input_dim, hidden_dim / 2, self.pc, ln)
            return (f, b)

        self.BOS = self.pc.add_parameters(input_dim)
        self.EOS = self.pc.add_parameters(input_dim)
        self.ROOT = self.pc.add_parameters(input_dim)

        self.layers = [_build_layer(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            self.layers.append(_build_layer(hidden_dim, hidden_dim))
        self.set_dropouts(input_dropout, output_dropout)

        self.spec = input_dim, hidden_dim, num_layers, input_dropout, output_dropout, ln

    def __call__(self, x):
        x = [dy.parameter(self.BOS), dy.parameter(self.ROOT)] + x + [dy.parameter(self.EOS)]
        h = self.transduce(x)
        h[:] = h[1:-1]
        return h

    def transduce(self, x):
        for (f,b) in self.layers:
            fs = f.initial_state().transduce(x)
            bs = b.initial_state().transduce(reversed(x))
            x = [dy.concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        return x

    def set_dropout(self, dropout):
        self.set_dropouts(dropout, dropout)

    def set_dropouts(self, input_dropout, output_dropout):
        for (f,b) in self.layers:
            f.set_dropouts(input_dropout, output_dropout)
            b.set_dropouts(input_dropout, output_dropout)
    
    def disable_dropout(self):
        for (f,b) in self.layers:
            f.disable_dropout()
            b.disable_dropout()

    def param_collection(self):
        return self.pc

    @staticmethod
    def from_spec(spec, model):
        input_dim, hidden_dim, num_layers, input_dropout, output_dropout, ln = spec
        return BiLSTM(model, input_dim, hidden_dim, num_layers, input_dropout, output_dropout, ln)
