import os
import logging
from logging import FileHandler
import random
import time
from datetime import timedelta
import dynet_config

from conllutils import DEPREL, FORM_NORM_CHARS, LEMMA_NORM_CHARS
from conllutils import shuffled_stream, read_conllu, create_dictionary, create_index, write_index, map_to_instances

from .utils import progressbar, open_file, UAS, LAS, EMS
from .word2vec import read_word2vec, index_word2vec

_MODEL_FILENAME="{0}model_{1}"

def train(model, trainer, params):
    best_epoch = 0
    best_score = None
    model.set_training(True)

    batch_size = params.batch_size
    total_size = len(params.train_data)
    if total_size % batch_size != 0:
        total_size += batch_size - (total_size % batch_size)

    pb = progressbar(total_size)

    for epoch in range(params.max_epochs):
        print(f"epoch {epoch + 1}")
        start_time = time.time()

        for step, batch in enumerate(shuffled_stream(params.train_data, batch_size=batch_size, total_size=total_size)):
            batch_loss = []
            batch_arc_error = 0
            batch_label_error = 0
            num_tokens = 0

            for example in batch:
                loss, arc_error, label_error = model.loss_and_errors(example)

                batch_loss.append(loss)
                batch_arc_error += arc_error
                batch_label_error += label_error

                num_tokens += len(example)
                pb.update(1)

            batch_loss = dy.esum(batch_loss)
            batch_loss_value = batch_loss.value()
            batch_loss.backward()
            trainer.update()
            dy.renew_cg()

            elapsed_time = time.time() - start_time
            params.logger.info("{0} {1} {2} {3} {4} {5} {6}".format(epoch + 1, step + 1, timedelta(seconds=elapsed_time),
                    num_tokens,
                    batch_loss_value / num_tokens,
                    batch_arc_error / num_tokens,
                    batch_label_error / num_tokens))

        model.save(_MODEL_FILENAME.format(params.model_basename, epoch))
        pb.finish()
        pb.reset()

        if params.validation_data:
            print(f"validating epoch {epoch + 1}")
            model.set_training(False)
            score = validate(model, params.validation_data)
            model.set_training(True)
            print(", ".join(str(metric) for metric in score))

            if best_score is None or best_score[1] < score[1]:
                best_epoch = epoch
                best_score = score
        else:
            best_epoch = epoch

    model.set_training(False)
    return best_epoch, best_score

def validate(model, validation_data, metrics=[UAS, LAS, EMS]):
    metrics = tuple(metric() for metric in metrics)

    pb = progressbar(len(validation_data))
    for gold in validation_data:
        parsed = model.parse(gold)
        for metric in metrics:
            metric(gold, parsed)
        pb.update(1)
    pb.finish()

    return metrics

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
        self.fields = tuple([f.lower() for f in self.fields])
        train_data = open_treebank(self.treebanks["train"], self.basename)
        print("building index...")
        self.index = create_index(create_dictionary(read_conllu(train_data), self.fields + (DEPREL, FORM_NORM_CHARS, LEMMA_NORM_CHARS)))
        write_index(self.model_basename, self.index)
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

        self.model_params["embeddings_dims"] = {f: (len(self.index[f]) + 1, dim) for (f, dim) in zip(self.fields, self.embeddings_dims)}
        self.model_params["labels_dim"] = len(self.index[DEPREL])

        from .parser import BiaffineParser
        model = BiaffineParser(pc, **self.model_params)
        if hasattr(self, "embeddings_vectors"):
            self._init_embeddings(model)

        return model

    def _config_trainer(self, pc):
        return dy.AdamTrainer(pc)

    def _init_embeddings(self, model):
        for (fs, fn) in self.embeddings_vectors.items():
            f = fs.lower()
            print(f"initializing {fs} embeddings...")
            vectors = index_word2vec(read_word2vec(open_embeddings(fn, self.basename)), self.index[f])
            num_init, num_vec = model.embeddings.embedding[f].init_from_vectors(vectors)
            print(f"initialized {num_init}/{num_vec} vectors")

    def _load_data(self, dataset):
        if dataset in self.treebanks:
            file = open_treebank(self.treebanks[dataset], self.basename)
            data = list(map_to_instances(read_conllu(file), self.index))
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

def main(params):
    pc = dy.ParameterCollection()
    model, trainer = params.config(pc)

    best_epoch, best_score = train(model, trainer, params)

    if best_score is not None:
        print(f"best epoch: {best_epoch} {best_score}")

    if params.test_data:
        if best_epoch > 0:
            pc = dy.ParameterCollection()
            model, = dy.load(_MODEL_FILENAME.format(params.model_basename, best_epoch), pc)
        score = validate(model, params.test_data)
        print(", ".join(str(metric) for metric in score))

params = Params({
    "basename" : "build/",
    "model_basename" : "build/en_ewt/",
    # "model_filename" : "build/en_ewt/model_1",
    "treebanks" : {"train": "en_ewt-ud-train.conllu", "dev": "en_ewt-ud-dev.conllu", "test": "en_ewt-ud-test.conllu"},
    "fields" : ("FORM_NORM", "UPOS_FEATS"),
    "embeddings_dims" : (100, 100),
    # "embeddings_vectors": {"FORM_NORM": "vectors_form_norm.txt"},
    "lstm_num_layers": 3,
    "lstm_dim": 800,
    "arc_mlp_dim": 500,
    "label_mlp_dim": 100,
    "input_dropouts": 0.33,
    "embeddings_dropout": 0.33,
    "lstm_dropout": 0.33,
    "arc_mlp_dropout": 0.33,
    "label_mlp_dropout": 0.33,
    "max_epochs" : 1,
    "batch_size": 10,
    "loss": "crossentropy",
    "random_seed" : 123456789,
    "dynet_mem" : 1024
})

params.dynet_config()
import dynet as dy
main(params)