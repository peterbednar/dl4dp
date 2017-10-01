from __future__ import print_function

import dynet_config
dynet_config.set(mem=1024,random_seed=123456789)
import dynet as dy
import numpy as np
import sys
from utils import FORM, XPOS, DEPREL
from utils import create_dictionary, create_index, read_conllu, map_to_instances, shuffled_stream
from utils import parse_projective
from layers import Embeddings, BiLSTM
import random

class MLP(object):
    pass

if __name__ == "__main__":
    random.seed(1)
    train_file = "../treebanks/train/en/en.conllu"
    index = create_index(create_dictionary(read_conllu(train_file)))
    train_data = list(map_to_instances(read_conllu(train_file), index, (FORM, XPOS)))

    max_epochs = 30
    lstm_dim = 250
    arc_hidden_dim = 100
    label_hidden_dim = 100

    pc = dy.ParameterCollection()
    # embeddings = Embeddings(pc, [(len(index[FORM])+1, 100), (len(index[XPOS])+1, 25)])
    # input_dim = embeddings.dim

    input_dim = 125
    num_labels = len(index[DEPREL])

    form_embeddings = pc.add_lookup_parameters((len(index[FORM])+1, 100))
    pos_embeddings = pc.add_lookup_parameters((len(index[XPOS])+1, 25))

    bilstm = BiLSTM(pc, input_dim, lstm_dim, 2)

    arc_mlp = MLP()
    label_mlp = MLP()

    arc_mlp.WH = pc.add_parameters((arc_hidden_dim, lstm_dim))
    arc_mlp.WD = pc.add_parameters((arc_hidden_dim, lstm_dim))
    arc_mlp.b1 = pc.add_parameters((arc_hidden_dim))
    arc_mlp.b2 = pc.add_parameters((1, arc_hidden_dim))

    label_mlp.WH = pc.add_parameters((label_hidden_dim, lstm_dim))
    label_mlp.WD = pc.add_parameters((label_hidden_dim, lstm_dim))
    label_mlp.b1 = pc.add_parameters((label_hidden_dim))
    label_mlp.b2 = pc.add_parameters((num_labels, label_hidden_dim))

    def predict_arc(head, dep, h, WH, WD):
        if WH[head] == None:
            WH[head] = dy.parameter(arc_mlp.WH) * h[head]
        if WD[dep] == None:
            WD[dep] = dy.parameter(arc_mlp.WD) * h[dep]

        b1 = dy.parameter(arc_mlp.b1)
        b2 = dy.parameter(arc_mlp.b2)
        W = WH[head] + WD[dep]
        return b2 * dy.tanh(W + b1)

    def predict_label(head, dep, h, WH, WD):
        if WH[head] == None:
            WH[head] = dy.parameter(label_mlp.WH) * h[head]
        if WD[dep] == None:
            WD[dep] = dy.parameter(label_mlp.WD) * h[dep]

        b1 = dy.parameter(label_mlp.b1)
        b2 = dy.parameter(label_mlp.b2)
        W = WH[head] + WD[dep]
        return b2 * dy.tanh(W + b1)

    def predict_arcs(h):
        num_nodes = len(h)
        WH = [None] * num_nodes
        WD = [None] * num_nodes
        return [[predict_arc(head, dep, h, WH, WD) for head in range(num_nodes)] for dep in range(num_nodes-1)]

    def predict_labels(h, heads):
        num_nodes = len(h)
        WH = [None] * num_nodes
        WD = [None] * num_nodes
        return [predict_label(heads[i], i+1, h, WH, WD) for i in range(num_nodes-1)]

    def embeddings(feats):
        x = []
        num_tokens, num_feats = feats.shape
        for i in range(num_tokens):
            xform = form_embeddings[feats[i,0]]
            xpos = pos_embeddings[feats[i,1]]
            x.append(dy.concatenate([xform, xpos]))
        return x

    def to_nparray(exprs):
        return np.array([exp.value() for exp in exprs])

    trainer = dy.AdamTrainer(pc)

    total_loss = 0.0
    total_arc_error = 0.0
    total_label_error = 0.0
    num_tokens = 0

    step = 0
    epoch = 0
    for example in shuffled_stream(train_data):
        arc_errors = 0
        arc_loss = []
        label_errors = 0
        label_loss = []

        x = embeddings(example.feats)
        s = bilstm(x)

        arc_scores = predict_arcs(s)
        label_scores = predict_labels(s, example.heads)

        for i in range(len(s)-1):
            scarray = to_nparray(arc_scores[i])
            gold = example.heads[i]
            best_wrong = max([(j, sc) for j, sc in enumerate(scarray) if j != gold], key=lambda x: x[1])[0]
            if scarray[gold] < scarray[best_wrong]:
                arc_errors += 1
            if scarray[gold] < scarray[best_wrong] + 1.0:
                arc_loss.append(arc_scores[i][best_wrong] - arc_scores[i][gold] + 1.0)

        for i in range(len(s)-1):
            scarray = to_nparray(label_scores[i])
            gold = example.labels[i] - 1
            best_wrong = max([(j, sc) for j, sc in enumerate(scarray) if j != gold], key=lambda x: x[1])[0]
            if scarray[gold] < scarray[best_wrong]:
                label_errors += 1
            if scarray[gold] < scarray[best_wrong] + 1.0:
                label_loss.append(label_scores[i][best_wrong] - label_scores[i][gold] + 1.0)

        loss = arc_loss + label_loss
        if loss:
            loss = dy.esum(loss)
            total_loss += loss.value()
            loss.backward()
            trainer.update()

        dy.renew_cg()

        total_arc_error += arc_errors
        total_label_error += label_errors
        num_tokens += len(example)

        step += 1
        if (step % 100) == 0:
            print("{0} {1} {2}".format(total_loss / num_tokens, total_arc_error / num_tokens, total_label_error / num_tokens))
            sys.stdout.flush()
            total_loss = 0.0
            total_arc_error = 0.0
            total_label_error = 0.0
            num_tokens = 0

        if (step % len(train_data)) == 0:
            epoch += 1
            print("epoch: {0}".format(epoch))
            if epoch >= max_epochs:
                break
