from __future__ import print_function

_rand_seed = 123456789
import dynet_config
dynet_config.set(mem=1024,random_seed=_rand_seed)
import random
random.seed(_rand_seed)

import sys
import os
import logging
from logging import FileHandler, Formatter
import dynet as dy
import numpy as np
import time
from datetime import timedelta
from collections import Counter
from models import MLPParser, BiaffineParser
from utils import extract_treebank
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
    print("UAS: {0:.4}, LAS: {1:.4}".format(uas, las))
    return uas, las

_MODEL_FILENAME="{0}model_{1}"

def train(model, trainer, params):
    log = logging.getLogger("dl4dp.train")
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
    train_len = len(params.train_data)
    pb = progressbar(train_len)

    for example in shuffled_stream(params.train_data):
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
            log.info("{0} {1} {2} {3} {4} {5}".format(epoch + 1, step, timedelta(seconds=elapsed_time),
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

            dy.save(_MODEL_FILENAME.format(params.model_basename, epoch), [model])

            if params.validation_data:
                print("validating epoch {0}".format(epoch))
                model.disable_dropout()
                score = validate(model, params.validation_data)
                model.enable_dropout()
                if best_score is None or best_score[1] < score[1]:
                    best_epoch = epoch
                    best_score = score
            else:
                best_epoch = epoch

            if epoch >= params.max_epochs:
                break

    model.disable_dropout()
    return best_epoch, best_score

class _input_dropout:

    def __init__(self, frequencies, dropout):
        self._frequencies = frequencies
        self._dropout = dropout

    def __call__(self, v, f):
        if self._dropout[f] > 0:
            drop = (random.random() < (self._dropout[f] / (self._dropout[f] + self._frequencies[f][v])))
            return 0 if drop else v
        else:
            return v

class Params:

    def __init__(self, entries):
        self.__dict__.update(entries)

    def config(self):
        self._basic_config()
        self._set_index()
        self._set_datasets()
        self._set_dims()
        self._set_input_dropout()
    
    def _basic_config(self):
        self.model_basename = self.basename + self.treebank + "/"
        if not os.path.isdir(self.model_basename):
            os.makedirs(self.model_basename)

        log = logging.getLogger("dl4dp.train")
        log.setLevel(logging.INFO)
        log.addHandler(FileHandler(self.model_basename + "train.log", mode="w"))

    def _set_index(self):
        self.fields = tuple([STR_TO_FIELD[f.lower()] for f in self.fields])

        train_data = extract_treebank(self.treebank, "train", self.basename)
        print("building index...")
        self.index = create_index(create_dictionary(read_conllu(train_data), self.fields + (DEPREL, )))
        print("building index done")

    def _set_datasets(self):
        self.train_data = self._load_data("train")
        self.validation_data = self._load_data("dev")
        self.test_data = self._load_data("test")

    def _set_dims(self):
        self.embeddings_dims = [(len(self.index[f])+1, dim) for (f, dim) in zip(self.fields, self.embeddings_dims)]
        self.labels_dim = len(self.index[DEPREL])
    
    def _set_input_dropout(self):
        # to be implemented
        pass

    def _load_data(self, dataset):
        file = extract_treebank(self.treebank, dataset, self.basename)
        if file:
            data = list(map_to_instances(read_conllu(file), self.index, self.fields))
            num_sentences = len(data)
            num_tokens = sum([len(tree) for tree in data])
            print("{0} sentences: {1}, tokens: {2}".format(dataset, num_sentences, num_tokens))
            return data
        else:
            return None

if __name__ == "__main__":

    params = Params({
        "basename" : "../build/",
        "treebank" : "en_ewt",
        "fields" : ("FORM_NORM", "UPOS_FEATS"),
        "embeddings_dims" : (100, 25),
        #"input_dropout" : (0.25, 0),
        "max_epochs" : 2
    })

    params.config()

    pc = dy.ParameterCollection()
    model = BiaffineParser(pc, **params.__dict__)
    #model = MLPParser(pc, **params.__dict__)
    trainer = dy.AdamTrainer(pc)

    best_epoch, best_score = train(model, trainer, params)

    if best_score is not None:
        print("best epoch: {0}, score: {1:.4} UAS, {2:.4} LAS".format(best_epoch, best_score[0], best_score[1]))

    if params.test_data:
        if best_epoch > 0:
            pc = dy.ParameterCollection()
            model, = dy.load(_MODEL_FILENAME.format(params.model_basename, best_epoch), pc)
        validate(model, params.test_data)
