from __future__ import print_function

import dynet_config
dynet_config.set(mem=1024,random_seed=12345)

import sys
import random
import dynet as dy
import numpy as np
from models import MLPParser
from utils import DepTree, map_to_instances, read_conllu, read_index

def arc_loss(model, tree):
    h = model.transduce(tree.feats)
    scores = model.predict_arcs(h)
    loss = [_hinge(sc, tree.heads[dep]) for dep, sc in enumerate(scores)]
    return dy.esum(loss)

def label_loss(model, tree):
    h = model.transduce(tree.feats)
    scores = model.predict_labels(tree.heads, h)
    loss = [_hinge(sc, tree.labels[dep] - 1) for dep, sc in enumerate(scores)]
    return dy.esum(loss)

def _hinge(exprs, target, margin=1.0):
    scores = exprs.value()
    best_wrong = max([(i, sc) for i, sc in enumerate(scores) if i != target], key=lambda x: x[1])[0]
    if scores[target] < scores[best_wrong] + margin:
        return exprs[best_wrong] - exprs[target] + margin
    else:
        return dy.constant(0.0)

def evaluate(model, validation_data):
    num_tokens = 0.
    correct_ua = correct_la = 0.

    model.disable_dropout()
    for i, gold in enumerate(validation_data):
        num_tokens += len(gold)
        parsed = model.parse(gold.feats)

        for n in range(len(gold)):
            if parsed.heads[n] == gold.heads[n]:
                correct_ua += 1.
                if parsed.labels[n] == gold.labels[n]:
                    correct_la += 1.

        if (i % 100) == 0:
            print(".", end="")
            sys.stdout.flush()
    model.enable_dropout()

    uas = correct_ua / num_tokens
    las = correct_la / num_tokens
    print("\nUAS: {0:.4}, LAS: {1:.4}".format(uas, las))

if __name__ == "__main__":
    basename = "../build/cs"
    index = read_index(basename)
    train_data = list(map_to_instances(read_conllu("../treebanks/train/cs/cs.conllu"), index))
    train_data = list(train_data[:10])

    pc = dy.ParameterCollection()
    model = MLPParser(pc, basename="../build/cs")
    model.enable_dropout()
    trainer = dy.AdamTrainer(pc)

    print("training examples: {0}".format(len(train_data)))
    for epoch in range(10):
        random.shuffle(train_data)

        total_loss = 0
        for i, tree in enumerate(train_data):
            dy.renew_cg()
            loss = arc_loss(model, tree)
            total_loss += loss.value()
            loss.backward()
            trainer.update()

            if (i % 100) == 0:
                print(".", end="")
                sys.stdout.flush()

        print("\naverage loss: {0}".format(total_loss / len(train_data)))
        evaluate(model, train_data)
