import math
import random
import dynet as dy
from collections import OrderedDict

class Embedding(object):

    def __init__(self, pc, dims, dropout=0, input_dropout=0, update=True):
        self.set_training(False)
        self.lookup = pc.add_lookup_parameters(dims)
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.update = update

    def __call__(self, index):
        if self.training and self.input_dropout > 0 and random.random() < self.input_dropout:
            index = 0
        x = dy.lookup(self.lookup, index, self.update)
        if self.training and self.dropout > 0:
            x = dy.dropout(x, self.dropout)
        return x

    def set_training(self, training):
        self.training = training

    def init_from_word2vec(self, vectors):
        num_init = 0
        num_rows = self.lookup.shape()[0]
        for (i, v) in vectors:
            self.lookup.init_row(i, v)
            num_init += 1
        return (num_init, num_rows)
    
class Embeddings(object):

    def __init__(self, model, field_dims, dropout=0, input_dropout=0, update=True):
        self.pc = model.add_subcollection()
        self.embedding = OrderedDict([(f, Embedding(self.pc, dims)) for f, dims in field_dims.items()])
        self.set_dropout(dropout, input_dropout)
        self.set_update(update)
        self.dim = sum([dim for (_,dim) in field_dims.values()])

    def __call__(self, instance):
        x = [[self.embedding[f](instance[f][i]) for f in self.embedding.keys()] for i in range(len(instance))]
        for i, v in enumerate(x):
            if len(v) > 1:
                x[i] = dy.concatenate(v)
            else:
                x[i] = v[0]
        return x

    def set_training(self, training):
        for embds in self.embedding.values():
            embds.set_training(training)

    def set_dropout(self, dropout, input_dropout=0):
        for f, embds in self.embedding.items():
            embds.dropout = dropout[f] if isinstance(dropout, dict) else dropout
            embds.input_dropout = input_dropout[f] if isinstance(input_dropout, dict) else input_dropout
    
    def set_update(self, update):
        for f, embds in self.embedding.items():
            embds.update = update[f] if isinstance(update, dict) else update
            
def leaky_relu(x):
    return dy.bmax(0.1 * x, x)

_STR_TO_ACT = {"tanh": dy.tanh, "logistic": dy.logistic, "relu": dy.rectify, "leaky_relu": leaky_relu}

class Dense(object):

    def __init__(self, model, input_dim, output_dim, act=dy.rectify, bias=True, init_gain=math.sqrt(2.), ln=False, dropout=0):
        self.set_training(False)
        if isinstance(act, str):
            act = _STR_TO_ACT[act]
        self.pc = model.add_subcollection()
        self.act = act
        self.ln = ln
        self.W = self.pc.add_parameters((output_dim, input_dim), init=dy.GlorotInitializer(gain=init_gain))
        if bias or ln:
            self.b = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(0.))
        else:
            self.b = None
        if ln:
            self.g = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(1.))
        self.dropout = dropout

    def __call__(self, x):
        if self.ln:
            y = dy.layer_norm(self.W * x, self.g, self.b)
        elif self.b is not None:
            y = dy.affine_transform([self.b, self.W, x])
        else:
            y = self.W * x
        if self.act is not None:
            y = self.act(y)
        if self.training and self.dropout > 0:
            y = dy.dropout(y, self.dropout)
        return y

    def set_training(self, training):
        self.training = training

class MultiLayerPerceptron(object):

    def __init__(self, model, input_dim, hidden_dim, output_dim, num_layers=1, act=dy.rectify, init_gain=math.sqrt(2.), ln=False, dropout=0):
        if isinstance(act, str):
            act = _STR_TO_ACT[act]
        self.pc = model.add_subcollection()
        self.dims = [input_dim] + [hidden_dim]*num_layers + [output_dim]
        self.layers = []
        for input_dim, output_dim in zip(self.dims, self.dims[1:-1]):
            self.layers.append(Dense(self.pc, input_dim, output_dim, act, init_gain, ln, dropout))
        self.layers.append(Dense(self.pc, self.dims[-2], self.dims[-1], act=None, bias=False, init_gain=init_gain))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def set_training(self, training):
        for layer in self.layers:
            layer.set_training(training)

class Bilinear(object):

    def __init__(self, model, dim):
        self.pc = model.add_subcollection()
        self.U = self.pc.add_parameters((dim, dim), init=dy.SaxeInitializer())

    def __call__(self, x, y):
        return dy.transpose(x) * self.U * y

class Biaffine(object):

    def __init__(self, model, input_dim, output_dim, bias_x=True, bias_y=True, bias=True):
        self.pc = model.add_subcollection()
        self.U = [Bilinear(model, input_dim) for _ in range(output_dim)]
        self.bias_x = self.pc.add_parameters((output_dim, input_dim), init=dy.ConstInitializer(.0)) if bias_x else None
        self.bias_y = self.pc.add_parameters((output_dim, input_dim), init=dy.ConstInitializer(.0)) if bias_y else None
        self.bias = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(.0)) if bias else None

    def __call__(self, x, y):
        z = dy.concatenate([u(x,y) for u in self.U])
        if self.bias_x is not None:
            z += self.bias_x * x
        if self.bias_y is not None:
            z += self.bias_y * y
        if self.bias is not None:
            z += self.bias
        return z

class BiLSTM(object):

    def __init__(self, model, input_dim, hidden_dim, num_layers=1, input_dropout=0, output_dropout=0, ln=False, boundary_tokens=False, root_token=False):
        self.pc = model.add_subcollection()
        self.dims = (input_dim, hidden_dim)
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout
        self.root_token = root_token
        self.boundary_tokens = boundary_tokens

        if boundary_tokens:
            self.BOS = self.pc.add_parameters(input_dim)
            self.EOS = self.pc.add_parameters(input_dim)
        if root_token:
            self.ROOT = self.pc.add_parameters(input_dim)

        def _build_layer(input_dim, hidden_dim, rnn_builder=dy.VanillaLSTMBuilder):
            f = rnn_builder(1, input_dim, hidden_dim // 2, self.pc, ln)
            b = rnn_builder(1, input_dim, hidden_dim // 2, self.pc, ln)
            return (f, b)

        self.layers = [_build_layer(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            self.layers.append(_build_layer(hidden_dim, hidden_dim))
        self.set_training(False)

    def __call__(self, x):
        if self.root_token:
            x = [self.ROOT] + x
        if self.boundary_tokens:
            x = [self.BOS] + x + [self.EOS]
        h = self.transduce(x)
        if self.boundary_tokens:
            h[:] = h[1:-1]
        return h

    def transduce(self, x):
        for (f,b) in self.layers:
            fs = f.initial_state().transduce(x)
            bs = b.initial_state().transduce(reversed(x))
            x = [dy.concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        return x

    def set_training(self, training):
        for (f,b) in self.layers:
            if training:
                f.set_dropouts(self.input_dropout, self.output_dropout)
                b.set_dropouts(self.input_dropout, self.output_dropout)
            else:
                f.disable_dropout()
                b.disable_dropout()
