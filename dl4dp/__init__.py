import os
import time
import random
from datetime import timedelta
import logging
from logging import FileHandler

import torch
from torch.optim import Adam
import numpy as np

from conllutils import FORM_NORM, UPOS_FEATS, DEPREL
from conllutils import read_conllu, create_index, map_to_instances, shuffled_stream

from .model import BiaffineParser
from .utils import progressbar

def train(model, optimizer, params):

    batch_size = params.batch_size
    total_size = len(params.train_data)
    if total_size % batch_size != 0:
        total_size += batch_size - (total_size % batch_size)

    pb = progressbar(total_size)

    step = 0
    for epoch in range(params.max_epoch):
        print(f"epoch: {epoch + 1}/{params.max_epoch}")

        start_time = time.time()
        for batch in shuffled_stream(params.train_data, total_size=total_size, batch_size=batch_size):
            optimizer.zero_grad()
            arc_loss, label_loss, arc_error, label_error = model.loss(batch)

            loss = arc_loss + label_loss
            loss.backward()
            optimizer.step()

            step += 1
            pb.update(batch_size)
            elapsed_time = time.time() - start_time
            num_words = sum([instance.length for instance in batch])

            params.logger.info(f"{epoch + 1} {step} {timedelta(seconds=elapsed_time)} {num_words} " +
                    f"{arc_loss.item()} {label_loss.item()} " +
                    f"{arc_error} {label_error}")

        pb.finish()
        pb.reset()

        torch.save(model, params.model_basename + 'model.pth')

        if params.validation_data:
            model.eval()
            uas, las = validate(model, params)
            print(f"UAS: {uas}, LAS: {las}")
            model.train()

def validate(model, params):
    batch_size = params.batch_size
    validation_data = params.validation_data

    uas_correct = 0
    las_correct = 0
    total_count = 0

    pb = progressbar(len(validation_data))

    for index in range(0, len(validation_data), batch_size):
        batch = validation_data[index:index + batch_size]
        arcs_pred, labels_pred = model.parse(batch)

        i = 0
        for instance in batch:
            for j in range(instance.length):
                if instance.head[j] == arcs_pred[i]:
                    uas_correct += 1
                    if instance.deprel[j] == labels_pred[i]:
                        las_correct += 1
                total_count += 1
                i += 1
            pb.update()

    pb.finish()

    uas = uas_correct / total_count
    las = las_correct / total_count
    return uas, las

class Params(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._basic_config()

    def _basic_config(self):
        os.makedirs(self.model_basename, exist_ok=True)

        log = logging.getLogger('dl4dp.train')
        log.setLevel(logging.INFO)
        log.addHandler(FileHandler(self.model_basename + 'train.log', mode='w'))
        self.logger = log

def debug(model, params):
    model = torch.load(params.model_basename + 'model.pth')
    model.eval()
    uas, las = validate(model, params)
    print(f"{uas} {las}")

def main():
    np.random.seed(1)
    torch.manual_seed(1)

    params = Params(max_epoch=1, batch_size=100, model_basename='build/en_ewt/')
    train_data = 'build/en_ewt-ud-train.conllu'
    validation_data = 'build/en_ewt-ud-dev.conllu'
    test_data = 'build/en_ewt-ud-test.conllu'

    print('building index...')
    index = create_index(
        read_conllu(train_data, skip_empty=True, skip_multiword=True), 
        fields={FORM_NORM, UPOS_FEATS, DEPREL})

    print('building index done')
    params.train_data = list(map_to_instances(
        read_conllu(train_data, skip_empty=True, skip_multiword=True),
        index))

    params.validation_data = list(map_to_instances(
        read_conllu(validation_data, skip_empty=True, skip_multiword=True),
        index))

    params.test_data = list(map_to_instances(
        read_conllu(test_data, skip_empty=True, skip_multiword=True),
        index))

    embedding_dims = {FORM_NORM: (len(index[FORM_NORM]) + 1, 100), UPOS_FEATS: (len(index[UPOS_FEATS]) + 1, 100)}
    labels_dim = len(index[DEPREL]) + 1
    model = BiaffineParser(embedding_dims, labels_dim)

    optimizer = Adam(model.parameters())
    train(model, optimizer, params)
    #debug(model, params)
