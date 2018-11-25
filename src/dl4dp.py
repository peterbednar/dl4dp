from __future__ import print_function

_rand_seed = 123456789
import dynet_config
dynet_config.set(mem=1024,random_seed=_rand_seed)
import random
random.seed(_rand_seed)

import sys
import os
import logging
import dynet as dy
import numpy as np
import time
from datetime import timedelta
from collections import Counter
from models import MLPParser, BiaffineParser
from utils import load_treebank
from utils import create_index, create_dictionary, DEPREL, STR_TO_FIELD
from utils import DepTree, map_to_instances, read_conllu, shuffled_stream
from utils import progressbar

def hinge_loss(scores, gold):
    error = 0
    loss = None
    gold = np.asscalar(gold)
    scarray = scores.npvalue()
    best_wrong = max([(i, sc) for i, sc in enumerate(scarray) if i != gold], key=lambda x: x[1])[0]
    if scarray[gold] < scarray[best_wrong]:
        error = 1
    if scarray[gold] < scarray[best_wrong] + 1.0:
        loss = scores[best_wrong] - scores[gold] + 1.0
    return error, loss

def validate(model, validation_data):
    num_tokens = 0
    correct_ua = correct_la = 0
    pb = progressbar(len(validation_data))

    for i, gold in enumerate(validation_data):
        num_tokens += len(gold)
        parsed = model.parse(gold.feats)
        for n in range(len(gold)):
            if parsed.heads[n] == gold.heads[n]:
                correct_ua += 1
                if parsed.labels[n] == gold.labels[n]:
                    correct_la += 1
        pb.update(1)
    pb.finish()

    uas = float(correct_ua) / num_tokens
    las = float(correct_la) / num_tokens
    print("uas: {0:.4}, las: {1:.4}".format(uas, las))
    return uas, las

_MODEL_FILENAME="{0}model_{1}"

def train(model, trainer, train_data, validation_data=None, max_epochs=30):
    model.enable_dropout()

    step_loss = 0.0
    step_arc_error = 0.0
    step_label_error = 0.0
    num_tokens = 0

    best_epoch = 0
    best_score = None

    step = 0
    epoch = 0
    dy.renew_cg()
    start_time = time.time()
    train_len = len(train_data)
    pb = progressbar(train_len)

    for example in shuffled_stream(train_data):
        if (step % train_len) == 0:
            print("epoch {0}".format(epoch + 1))

        loss = []
        h = model.transduce(example.feats)

        arc_scores = model.predict_arcs(h)
        for i in range(len(example)):
            arc_error, arc_loss = hinge_loss(arc_scores[i], example.heads[i])
            step_arc_error += arc_error
            if arc_loss:
                loss.append(arc_loss)

        label_scores = model.predict_labels(example.heads, h)
        for i in range(len(example)):
            label_error, label_loss = hinge_loss(label_scores[i], example.labels[i] - 1)
            step_label_error += label_error
            if label_loss:
                loss.append(label_loss)

        if loss:
            loss = dy.esum(loss)
            step_loss += loss.value()
            loss.backward()
            trainer.update()
        dy.renew_cg()

        num_tokens += len(example)
        step += 1
        pb.update(1)

        if (step % 100) == 0:
            elapsed_time = time.time() - start_time
            logging.info("{0} {1} {2} {3} {4} {5}".format(epoch + 1, step, timedelta(seconds=elapsed_time),
                    step_loss / num_tokens,
                    step_arc_error / num_tokens,
                    step_label_error / num_tokens))
            step_loss = 0.0
            step_arc_error = 0.0
            step_label_error = 0.0
            num_tokens = 0

        if (step % train_len) == 0:
            epoch += 1
            start_time = time.time()
            pb.finish()
            pb.reset()

            dy.save(_MODEL_FILENAME.format(basename, epoch), [model])

            if validation_data:
                print("validating epoch {0}".format(epoch))
                model.disable_dropout()
                score = validate(model, validation_data)
                model.enable_dropout()
                if best_score is None or best_score[1] < score[1]:
                    best_epoch = epoch
                    best_score = score
            else:
                best_epoch = epoch

            if epoch >= max_epochs:
                break

    model.disable_dropout()
    return best_epoch, best_score

def _load_data(treebank, dataset, index, fields, basename):
    file = load_treebank(treebank, dataset, basename)
    if file:
        data = list(map_to_instances(read_conllu(file), index, fields))
        num_sentences = len(data)
        num_tokens = sum([len(tree) for tree in data])
        print("{0} sentences: {1}, tokens: {2}".format(dataset, num_sentences, num_tokens))
        return data
    else
        return None

def _index_frequencies(dic, index, fields):
    count = tuple([Counter() for f in fields])
    for i, f in enumerate(fields):
        for v, freq in dic[f].items():
            count[i][index[f][v]] += freq
    return count

if __name__ == "__main__":
    basename = "../build/"
    treebank = "en_ewt"
    fields = ("FORM_NORM", "UPOS_FEATS")
    embeddings_dims = (100, 25)
    input_dropout = (0.25, 0)
    max_epochs = 2

    if not os.path.isdir(basename):
        os.makedirs(basename)

    logging.basicConfig(filename=basename + "train.log", filemode="w", format="%(message)s", level=logging.INFO)

    fields = tuple([STR_TO_FIELD[f.lower()] for f in fields])

    print("building index...")
    dic = create_dictionary(read_conllu(load_treebank(treebank, "train", basename)), fields + (DEPREL, ))
    index = create_index(dic)
    print("building index done")
    train_data = _load_data(treebank, "train", index, fields, basename)
    validation_data = _load_data(treebank, "dev", index, fields, basename)

    embeddings_dims = [(len(index[f])+1, dim) for (f, dim) in zip(fields, embeddings_dims)]
    labels_dim = len(index[DEPREL])
    if input_dropout is not None:
        frequencies = _index_frequencies(dic, index, fields)
        dropout = input_dropout
        def _input_dropout(v, f):
            if dropout[f] > 0:
                drop = (random.random() < (dropout[f] / (dropout[f] + frequencies[f][v])))
                return 0 if drop else v
            else:
                return v
        input_dropout = _input_dropout

    pc = dy.ParameterCollection()
    model = BiaffineParser(pc, embeddings_dims=embeddings_dims, labels_dim=labels_dim, input_dropout=input_dropout)
    #model = MLPParser(pc, embeddings_dims=embeddings_dims, labels_dim=labels_dim, input_dropout=input_dropout)
    trainer = dy.AdamTrainer(pc)

    best_epoch, best_score = train(model, trainer, train_data, validation_data, max_epochs)

    if best_score is not None:
        print("best epoch: {0}, score: {1:.4} uas, {2:.4} las".format(best_epoch, best_score[0], best_score[1]))

    test_data = _load_data(treebank, "test", index, fields, basename)
    if test_data:
        if best_epoch > 0:
            pc = dy.ParameterCollection()
            model, = dy.load(_MODEL_FILENAME.format(basename, best_epoch), pc)
        validate(model, test_data)
