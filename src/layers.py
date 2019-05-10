import math
import random
import dynet as dy

class Embeddings(object):

    def __init__(self, model, dims, dropout=0, input_dropout=0, update=True):
        self.pc = model.add_subcollection()
        self.lookup = [self.pc.add_lookup_parameters(dim) for dim in dims]
        self.set_dropout(dropout, input_dropout)
        self.set_update(update)
        self.dim = sum([dim for (_,dim) in dims])

    def _lookup(self, feats, i, f):
        v = feats[i,f]
        input_dropout = self.input_dropout[f]
        if input_dropout > 0 and random.random() < input_dropout:
            v = 0
        embds = dy.lookup(self.lookup[f], v, update=self.update[f])
        dropout = self.dropout[f]
        if dropout > 0:
            embds = dy.dropout(embds, dropout)
        return embds

    def __call__(self, feats):
        num_tokens, num_feats = feats.shape
        if num_feats > 1:
            x = [dy.concatenate([self._lookup(feats,i,f) for f in range(num_feats)]) for i in range(num_tokens)]
        else:
            x = [self._lookup(feats,i,0) for i in range(num_tokens)]
        return x

    def set_dropout(self, dropout, input_dropout=0):
        self.input_dropout = input_dropout if isinstance(input_dropout, (tuple, list)) else [input_dropout] * len(self.lookup)
        self.dropout = dropout if isinstance(dropout, (tuple, list)) else [dropout] * len(self.lookup)
    
    def disable_dropout(self):
        self.set_dropout(0, 0)

    def set_update(self, update):
        self.update = update if isinstance(update, (tuple, list)) else [update] * len(self.lookup)

    def init_from_word2vec(self, feature, vectors):
        lookup = self.lookup[feature]
        num_init = 0
        num_rows = lookup.shape()[0]
        for (i, v) in vectors:
            lookup.init_row(i, v)
            num_init += 1
        return (num_init, num_rows)
            
def leaky_relu(x):
    return dy.bmax(0.1 * x, x)

_STR_TO_ACT = {"tanh": dy.tanh, "logistic": dy.logistic, "relu": dy.rectify, "leaky_relu": leaky_relu}

class Dense(object):

    def __init__(self, model, input_dim, output_dim, act=dy.rectify, init_gain=math.sqrt(2.), ln=False, dropout=0):
        if isinstance(act, str):
            act = _STR_TO_ACT[act]
        self.pc = model.add_subcollection()
        self.act = act
        self.ln = ln
        self.W = self.pc.add_parameters((output_dim, input_dim), init=dy.GlorotInitializer(gain=init_gain))
        self.b = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(0.))
        if ln:
            self.g = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(1.))
        self.dropout = dropout

    def __call__(self, x):
        if self.ln:
            y = dy.layer_norm(self.W * x, self.g, self.b)
        else:
            y = dy.affine_transform([self.b, self.W, x])
        y = self.act(y)
        if self.dropout > 0:
            y = dy.dropout(y, self.dropout)
        return y

    def set_dropout(self, dropout):
        self.dropout = dropout

    def disable_dropout(self):
        self.set_dropout(0)

class Identity(object):

    def __init__(self, model, input_dim, output_dim, init_gain=math.sqrt(2.), dropout=0):
        self.pc = model.add_subcollection()
        self.W = self.pc.add_parameters((output_dim, input_dim), init=dy.GlorotInitializer(gain=init_gain))
        self.dropout = dropout
    
    def __call__(self, x):
        y = self.W * x
        if self.dropout > 0:
            y = dy.dropout(y, self.dropout)
        return y

    def set_dropout(self, dropout):
        self.dropout = dropout

    def disable_dropout(self):
        self.set_dropout(0)

class MultiLayerPerceptron(object):

    def __init__(self, model, input_dim, hidden_dim, output_dim, num_layers=1, act=dy.rectify, init_gain=math.sqrt(2.), ln=False, dropout=0):
        if isinstance(act, str):
            act = _STR_TO_ACT[act]
        self.pc = model.add_subcollection()
        self.dims = [input_dim] + [hidden_dim]*num_layers + [output_dim]
        self.layers = []
        for input_dim, output_dim in zip(self.dims, self.dims[1:-1]):
            self.layers.append(Dense(self.pc, input_dim, output_dim, act, init_gain, ln))
        self.layers.append(Identity(self.pc, self.dims[-2], self.dims[-1], init_gain))
        self.set_dropout(dropout)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def set_dropout(self, dropout):
        for layer in self.layers:
            layer.set_dropout(dropout)
    
    def disable_dropout(self):
        for layer in self.layers:
            layer.disable_dropout()

class Bilinear(object):

    def __init__(self, model, dim):
        self.pc = model.add_subcollection()
        self.U = self.pc.add_parameters((dim, dim), init=dy.SaxeInitializer())

    def __call__(self, x, y):
        return dy.transpose(x) * self.U * y

class Biaffine(object):

    def __init__(self, model, input_dim, output_dim):
        self.pc = model.add_subcollection()
        self.U = [Bilinear(model, input_dim) for _ in range(output_dim)]
        self.x_bias = self.pc.add_parameters((output_dim, input_dim), init=dy.ConstInitializer(.0))
        self.y_bias = self.pc.add_parameters((output_dim, input_dim), init=dy.ConstInitializer(.0))
        self.bias = self.pc.add_parameters(output_dim, init=dy.ConstInitializer(.0))

    def __call__(self, x, y):
        xUy = dy.concatenate([u(x,y) for u in self.U])
        return xUy + self.x_bias * x + self.y_bias * y + self.bias 

class BiLSTM(object):

    def __init__(self, model, input_dim, hidden_dim, num_layers=1, input_dropout=0, output_dropout=0, ln=False, boundary_tokens=True, root_token=True):
        self.pc = model.add_subcollection()
        self.dims = (input_dim, hidden_dim)
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
        self.set_dropouts(input_dropout, output_dropout)

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
