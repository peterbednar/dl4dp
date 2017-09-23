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
    loss = [dy.hinge(sc, tree.heads[dep], 1.0) for dep, sc in enumerate(scores)]
    return dy.esum(loss)

def label_loss(model, tree):
    h = model.transduce(tree.feats)
    scores = model.predict_labels(tree.heads, h)
    loss = [dy.hinge(sc, tree.labels[dep] - 1, 1.0) for dep, sc in enumerate(scores)]
    return dy.esum(loss)

if __name__ == "__main__":
    basename = "../build/cs"
    index = read_index(basename)
    train_data = list(map_to_instances(read_conllu("../treebanks/train/cs/cs.conllu"), index))

    pc = dy.ParameterCollection()
    model = MLPParser(pc, basename="../build/cs")
    trainer = dy.AdamTrainer(pc)

    print("training examples: {0}".format(len(train_data)))
    for epoch in range(100):
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
