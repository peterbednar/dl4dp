from __future__ import print_function

import dynet_config
dynet_config.set(mem=1024,random_seed=12345)

import sys
import random
import dynet as dy
import numpy as np
from models import MLPParser, FIELDS
from utils import DepTree, map_to_instances, read_conllu, read_index

def arc_loss(model, tree):
    h = model.transduce(tree.feats)
    scores = model.predict_arcs(h)
    loss = [hinge_loss(sc, tree.heads[dep]) for dep, sc in enumerate(scores)]
    return dy.esum(loss)

def label_loss(model, tree):
    h = model.transduce(tree.feats)
    scores = model.predict_labels(tree.heads, h)
    loss = [hinge_loss(sc, tree.labels[dep] - 1) for dep, sc in enumerate(scores)]
    return dy.esum(loss)

def hinge_loss(exprs, target, margin=1.0):
    scores = exprs.value()
    best_wrong = max([(i, sc) for i, sc in enumerate(scores) if i != target], key=lambda x: x[1])[0]
    if scores[target] < scores[best_wrong] + margin:
        return exprs[best_wrong] - exprs[target] + margin
    else:
        return dy.zeros(1)

def shuffled_stream(data):
    while True:
        random.shuffle(data)
        for d in data:
            yield d

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

    basename = "../build/en"
    index = read_index(basename)
    train_data = list(map_to_instances(read_conllu("../treebanks/train/en/en.conllu"), index, FIELDS))

    pc = dy.ParameterCollection()
    model = MLPParser(pc, basename="../build/en")
    model.enable_dropout()
    trainer = dy.AdamTrainer(pc)

    print("training sentences: {0}, tokens: {1}".format(len(train_data), sum([len(tree) for tree in train_data])))

    batch_size = 50
    max_steps = 1000

    step = 0
    total_loss = 0
    batch_loss = []
    batch_tokens = 0

    dy.renew_cg()
    for tree in shuffled_stream(train_data):

        batch_loss.append(arc_loss(model, tree))
        batch_loss.append(label_loss(model, tree))

        batch_tokens += len(tree)
        if batch_tokens >= batch_size:
            loss = dy.esum(batch_loss) * (1.0 / batch_tokens)
            total_loss += loss.value()
            loss.backward()
            trainer.update()

            dy.renew_cg()
            batch_loss = []
            batch_tokens = 0
            step += 1

            if (step % 100) == 0:
                print(".", end="")
                sys.stdout.flush()

            if (step % 1000) == 0:
                print("\naverage loss: {0}".format(total_loss / 1000))
                evaluate(model, train_data)
                total_loss = 0.0

            if step >= max_steps:
                break
