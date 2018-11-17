from __future__ import print_function

_rand_seed = 123456789
import dynet_config
dynet_config.set(mem=1024,random_seed=_rand_seed)
import random
random.seed(_rand_seed)

import sys
import os
import dynet as dy
import numpy as np
import time
from datetime import timedelta
from collections import Counter
from models import MLPParser, BiaffineParser
from utils import create_index, create_dictionary, FORM_NORM, UPOS_FEATS, DEPREL, STR_TO_FIELD
from utils import DepTree, map_to_instances, read_conllu, shuffled_stream, count_frequency

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

    uas = float(correct_ua) / num_tokens
    las = float(correct_la) / num_tokens
    print("\nuas: {0:.4}, las: {1:.4}".format(uas, las))
    return uas, las

_MODEL_FILENAME="{0}_model_{1}"

def train(model, trainer, train_data, validation_data=None, max_epochs=30):
    model.enable_dropout()

    step_loss = 0.0
    step_arc_error = 0.0
    step_label_error = 0.0
    num_tokens = 0

    best_epoch = 0
    best_score = (0,0)

    step = 0
    epoch = 0
    dy.renew_cg()
    start_time = time.time()
    for example in shuffled_stream(train_data):
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
        if (step % 100) == 0:
            elapsed_time = time.time() - start_time
            print("{0} {1} {2} {3} {4}".format(step, timedelta(seconds=elapsed_time),
                    step_loss / num_tokens,
                    step_arc_error / num_tokens,
                    step_label_error / num_tokens))
            sys.stdout.flush()
            step_loss = 0.0
            step_arc_error = 0.0
            step_label_error = 0.0
            num_tokens = 0

        if (step % len(train_data)) == 0:
            epoch += 1
            print("epoch: {0}".format(epoch))
            if (epoch % 1) == 0 and validation_data:
                model.disable_dropout()
                score = validate(model, validation_data)
                model.enable_dropout()
                if best_score[1] < score[1]:
                    best_epoch = epoch
                    best_score = score
            dy.save(_MODEL_FILENAME.format(basename, epoch), [model])
            start_time = time.time()
            if epoch >= max_epochs:
                break

    model.disable_dropout()
    return best_epoch, best_score

def _load_data(filename, index, fields, msg=None):
    data = list(map_to_instances(read_conllu(filename), index, fields))
    if msg is not None:
        num_sentences = len(data)
        num_tokens = sum([len(tree) for tree in data])
        print(msg.format(num_sentences, num_tokens))
    return data

def _index_frequencies(dic, index, fields):
    count = tuple([Counter() for f in fields])
    for i, f in enumerate(fields):
        for v, freq in dic[f].items():
            count[i][index[f][v]] += freq
    return count

if __name__ == "__main__":
    lang = "en"
    treebank = "en"
    max_epochs = 1

    fields = (FORM_NORM, UPOS_FEATS)
    embeddings_dims = (100, 25)
    input_dropout = (0.25, 0)
    
    basename = "../build/" + treebank
    train_filename = "../treebanks/train/" + lang + "/" + treebank + ".conllu"
    validation_filename = "../treebanks/dev/" + lang + "/" + treebank + ".conllu"
    test_filename = "../treebanks/test/" + lang + "/" + treebank + ".conllu"

    if not os.path.isdir(basename):
        os.makedirs(basename)

    print("building index...", end=" ")
    dic = create_dictionary(read_conllu(train_filename), fields + (DEPREL, ))
    index = create_index(dic)
    print("done")
    train_data = _load_data(train_filename, index, fields, "training sentences: {0}, tokens: {1}")
    validation_data = _load_data(validation_filename, index, fields, "validation sentences: {0}, tokens: {1}") if validation_filename is not None else None

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

    if best_epoch > 0:
        print("best epoch: {0}, score: {1:.4} uas, {2:.4} las".format(best_epoch, best_score[0], best_score[1]))
    if test_filename:
        test_data = _load_data(test_filename, index, fields, "testing sentences: {0}, tokens: {1}")
        if best_epoch > 0:
            pc = dy.ParameterCollection()
            model, = dy.load(_MODEL_FILENAME.format(basename, best_epoch), pc)
        validate(model, test_data)
