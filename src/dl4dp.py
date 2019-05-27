import os
import logging
from logging import FileHandler
import random
import time
import numpy as np
from datetime import timedelta
from utils import str_to_field
from utils import create_index, create_dictionary, DEPREL, write_index, open_file
from utils import read_conllu, map_to_instances, shuffled_stream
from utils import progressbar
from word2vec import read_word2vec, index_word2vec
import dynet_config

_MODEL_FILENAME="{0}model_{1}"

def train(model, trainer, params):

    def error_and_loss(scores, correct):
        pred = np.argmax(scores.npvalue())
        error = 1 if pred != correct else 0
        loss = params.loss(scores, correct)
        return error, loss

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
            print(f"epoch {epoch + 1}")

        loss = []
        h = model.transduce(example.feats)

        arc_scores = model.predict_arcs(h)
        for i in range(len(example)):
            arc_error, arc_loss = error_and_loss(arc_scores[i], example.heads[i])
            step_arc_error += arc_error
            if arc_loss:
                loss.append(arc_loss)

        label_scores = model.predict_labels(example.heads, h)
        for i in range(len(example)):
            label_error, label_loss = error_and_loss(label_scores[i], example.labels[i] - 1)
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
            params.logger.info("{0} {1} {2} {3} {4} {5} {6}".format(epoch + 1, step, timedelta(seconds=elapsed_time),
                    num_tokens,
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
                print(f"validating epoch {epoch}")
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

def validate(model, validation_data):
    num_tokens = 0
    correct_ua = correct_la = 0
    pb = progressbar(len(validation_data))

    for gold in validation_data:
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
    print(f"UAS: {uas:.4f}, LAS: {las:.4f}")
    return uas, las

class Params(object):

    def __init__(self, params):
        self.__dict__.update(params)
        self.model_params = dict(params)

    def config(self, pc):
        self._basic_config()
        self._set_index()
        self._set_datasets()
        return (self._config_model(pc), self._config_trainer(pc))

    def dynet_config(self):
        random_seed = getattr(self, "random_seed", 0)
        dynet_mem = getattr(self, "dynet_mem", 1024)
        dynet_config.set(mem=dynet_mem, random_seed=random_seed)
        random.seed(random_seed)

    def _basic_config(self):
        os.makedirs(self.model_basename, exist_ok=True)

        log = logging.getLogger("dl4dp.train")
        log.setLevel(logging.INFO)
        log.addHandler(FileHandler(self.model_basename + "train.log", mode="w"))
        self.logger = log

    def _set_index(self):
        self.fields = tuple([str_to_field(f) for f in self.fields])

        train_data = open_treebank(self.treebanks["train"], self.basename)
        print("building index...")
        self.index = create_index(create_dictionary(read_conllu(train_data), self.fields + (DEPREL, )))
        write_index(self.index, basename=self.model_basename)
        print("building index done")

    def _set_datasets(self):
        self.train_data = self._load_data("train")
        self.validation_data = self._load_data("dev")
        self.test_data = self._load_data("test")

    def _config_model(self, pc):
        if hasattr(self, "model_filename"):
            filename = getattr(self, "model_filename")
            print(f"loading model '{filename}'")
            model, = dy.load(filename, pc)
            return model

        self.model_params["embeddings_dims"] = [(len(self.index[f]) + 1, dim) for (f, dim) in zip(self.fields, self.embeddings_dims)]
        self.model_params["labels_dim"] = len(self.index[DEPREL])

        model = BiaffineParser(pc, **self.model_params)
        if hasattr(self, "embeddings_vectors"):
            self._init_embeddings(model)

        return model

    def _config_trainer(self, pc):
        loss = getattr(self, "loss", "crossentropy")
        if loss == "hinge":
            self.loss = dy.hinge
        elif loss == "crossentropy":
            self.loss = dy.pickneglogsoftmax
        else:
            raise ValueError(f"unknown loss function: {loss}")

        return dy.AdamTrainer(pc)

    def _init_embeddings(self, model):
        for (fs, fn) in self.embeddings_vectors.items():
            f = str_to_field(fs)
            fi = self.fields.index(f)

            print(f"initializing {fs} embeddings...")
            vectors = index_word2vec(read_word2vec(open_embeddings(fn, self.basename)), self.index[f])
            num_init, num_vec = model.embeddings.init_from_word2vec(fi, vectors)
            print(f"initialized {num_init}/{num_vec} vectors")

    def _load_data(self, dataset):
        if dataset in self.treebanks:
            file = open_treebank(self.treebanks[dataset], self.basename)
            data = list(map_to_instances(read_conllu(file), self.index, self.fields))
            num_sentences = len(data)
            num_tokens = sum([len(tree) for tree in data])
            print(f"{dataset} sentences: {num_sentences}, tokens: {num_tokens}")
            return data
        else:
            return None

def open_treebank(treebank, basename=""):
    filename = basename + treebank
    return open_file(filename)

def open_embeddings(embeddings, basename=""):
    filename = basename + embeddings
    return open_file(filename, errors="replace")

if __name__ == "__main__":

    params = Params({
        "basename" : "../build/",
        "model_basename" : "../build/en_ewt/",
        "model_filename" : "../build/en_ewt/model_1",
        "treebanks" : {"train": "en_ewt-ud-train.conllu", "dev": "en_ewt-ud-dev.conllu", "test": "en_ewt-ud-test.conllu"},
        "fields" : ("FORM_NORM", "UPOS_FEATS"),
        "embeddings_dims" : (100, 100),
        "embeddings_vectors": {"FORM_NORM": "vectors_form_norm.txt"},
        "lstm_num_layers": 3,
        "lstm_dim": 400,
        "arc_mlp_dim": 100,
        "label_mlp_dim": 100,
        "input_dropouts": 0.33,
        "embeddings_dropout": 0.33,
        "lstm_dropout": 0.33,
        "arc_mlp_dropout": 0.33,
        "label_mlp_dropout": 0.33,
        "max_epochs" : 2,
        "loss": "crossentropy",
        "random_seed" : 123456789,
        "dynet_mem" : 1024
    })

    params.dynet_config()
    import dynet as dy
    from models import BiaffineParser

    pc = dy.ParameterCollection()
    model, trainer = params.config(pc)

    best_epoch, best_score = train(model, trainer, params)

    if best_score is not None:
        print(f"best epoch: {best_epoch}, score: {best_score[0]:.4f} UAS, {best_score[1]:.4f} LAS")

    if params.test_data:
        if best_epoch > 0:
            pc = dy.ParameterCollection()
            model, = dy.load(_MODEL_FILENAME.format(params.model_basename, best_epoch), pc)
        validate(model, params.test_data)
