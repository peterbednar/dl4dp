from __future__ import print_function

_rand_seed = 123456789
import dynet_config
dynet_config.set(mem=1024,random_seed=_rand_seed)
import random
random.seed(_rand_seed)

import sys
import dynet as dy
import numpy as np
from models import MLPParser
from utils import create_index, create_dictionary, FORM, XPOS, DEPREL
from utils import DepTree, map_to_instances, read_conllu, shuffled_stream

def get_arc_loss(tree, scores):
    error = 0
    loss = []

    for i in range(len(tree)):
        scarray = scores[i].npvalue()
        gold = tree.heads[i]
        #  or j != (i+1)
        best_wrong = max([(j, sc) for j, sc in enumerate(scarray) if j != gold], key=lambda x: x[1])[0]
        if scarray[gold] < scarray[best_wrong]:
            error += 1
        if scarray[gold] < scarray[best_wrong] + 1.0:
            loss.append(arc_scores[i][best_wrong] - arc_scores[i][gold] + 1.0)

    return error, loss

def get_label_loss(tree, scores):
    error = 0
    loss = []

    for i in range(len(tree)):
        scarray = scores[i].npvalue()
        gold = tree.labels[i] - 1
        best_wrong = max([(j, sc) for j, sc in enumerate(scarray) if j != gold], key=lambda x: x[1])[0]
        if scarray[gold] < scarray[best_wrong]:
            error += 1
        if scarray[gold] < scarray[best_wrong] + 1.0:
            loss.append(label_scores[i][best_wrong] - label_scores[i][gold] + 1.0)

    return error, loss

def validate(model, validation_data):
    num_tokens = 0
    correct_ua = correct_la = 0

    model.disable_dropout()
    for i, gold in enumerate(validation_data):
        num_tokens += len(gold)
        parsed = model.parse(gold.feats)

        for n in range(len(gold)):
            if parsed.heads[n] == gold.heads[n]:
                correct_ua += 1
                if parsed.labels[n] == gold.labels[n]:
                    correct_la += 1

        if (i % 100) == 0:
            print(".", end="")
            sys.stdout.flush()
    model.enable_dropout()

    uas = float(correct_ua) / num_tokens
    las = float(correct_la) / num_tokens
    print("\nUAS: {0:.4}, LAS: {1:.4}".format(uas, las))

if __name__ == "__main__":
    max_epochs = 30

    basename = "../build/en"
    train_filename = "../treebanks/train/en/en.conllu"

    index = create_index(create_dictionary(read_conllu(train_filename), (FORM, XPOS, DEPREL)))
    train_data = list(map_to_instances(read_conllu("../treebanks/train/en/en.conllu"), index, (FORM, XPOS)))
    validation_data = train_data

    embeddings_dims = [(len(index[FORM])+1, 100), (len(index[XPOS])+1, 25)]
    labels_dim = len(index[DEPREL])

    pc = dy.ParameterCollection()
    model = MLPParser(pc, embeddings_dims=embeddings_dims, labels_dim=labels_dim)
    model.enable_dropout()
    trainer = dy.AdamTrainer(pc)

    print("training sentences: {0}, tokens: {1}".format(len(train_data), sum([len(tree) for tree in train_data])))

    step_loss = 0.0
    step_arc_error = 0.0
    step_label_error = 0.0
    num_tokens = 0

    step = 0
    epoch = 0
    dy.renew_cg()
    for example in shuffled_stream(train_data):
        loss = []
        h = model.transduce(example.feats)

        arc_scores = model.predict_arcs(h)
        arc_error, arc_loss = get_arc_loss(example, arc_scores)
        step_arc_error += arc_error
        loss.extend(arc_loss)

        label_scores = model.predict_labels(example.heads, h)
        label_error, label_loss = get_label_loss(example, label_scores)
        step_label_error += label_error
        loss.extend(label_loss)

        if loss:
            loss = dy.esum(loss)
            step_loss += loss.value()
            loss.backward()
            trainer.update()
        dy.renew_cg()

        num_tokens += len(example)
        step += 1
        if (step % 100) == 0:
            print("{0} {1} {2}".format(step_loss / num_tokens, step_arc_error / num_tokens, step_label_error / num_tokens))
            sys.stdout.flush()
            step_loss = 0.0
            step_arc_error = 0.0
            step_label_error = 0.0
            num_tokens = 0

        if (step % len(train_data)) == 0:
            dy.save(basename + "_model", [model])
            epoch += 1
            if (epoch % 1) == 0 and validation_data:
                validate(model, validation_data)
            print("epoch: {0}".format(epoch))
            if epoch >= max_epochs:
                break
