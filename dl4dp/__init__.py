import os
import time
import random
from datetime import timedelta
import logging
from logging import FileHandler

import torch
from torch.optim import Adam
import numpy as np

from conllutils import pipe

from .model import BiaffineParser
from .utils import progressbar

def train(model, optimizer, params):
    best_epoch = 0
    best_score = None

    batch_size = params.batch_size
    total_size = len(params.train_data)
    if total_size % batch_size != 0:
        total_size += batch_size - (total_size % batch_size)

    pb = progressbar(total_size)

    for epoch in range(params.max_epoch):
        print(f'epoch: {epoch + 1}/{params.max_epoch}')
        start_time = time.time()

        for step, batch in enumerate(pipe(params.train_data).stream(total_size).shuffle().batch(batch_size)):
            optimizer.zero_grad()
            loss, metrics = model.loss(batch)
            loss.backward()
            optimizer.step()

            pb.update(batch_size)
            elapsed_time = time.time() - start_time
            num_words = sum([instance.length for instance in batch])

            params.logger.info(f'{epoch + 1} {step + 1} {timedelta(seconds=elapsed_time)} {num_words} {loss.item()} ' +
                ' '.join([str(metric.item()) for metric in metrics]))

        torch.save(model, params.model_basename + f'model-{epoch + 1}.pth')
        pb.finish()
        pb.reset()

        if params.validation_data:
            print(f'validating epoch {epoch + 1}/{params.max_epoch}')
            score, metrics = validate(model, params.validation_data, params.batch_size)
            print(', '.join(f"{metric[0]}:{metric[1]:.4f}" for metric in metrics))

            if best_score is None or best_score < score:
                best_score = score
                best_epoch = epoch + 1
        else:
            best_epoch = epoch

    if best_score is not None:
        print(f'best epoch: {best_epoch}, score: {best_score:.4f}')

    return best_epoch, best_score

def validate(model, validation_data, batch_size=100):
    mode = model.training
    model.eval()

    total_size = len(validation_data)
    pb = progressbar(total_size)
    uas_correct = las_correct = em_correct = total = 0

    for batch in pipe(validation_data).batch(batch_size):
        arcs_pred, labels_pred = model.parse(batch)

        i = 0
        for instance in batch:
            match = True
            for j in range(instance.length):
                if instance.head[j] == arcs_pred[i]:
                    uas_correct += 1
                    if instance.deprel[j] == labels_pred[i]:
                        las_correct += 1
                    else:
                        match = False
                total += 1
                i += 1
            if match:
                em_correct += 1
            pb.update()

    pb.finish()

    uas = uas_correct / total
    las = las_correct / total
    em = em_correct / total_size
    metrics = (('UAS', uas), ('LAS', las), ('EM', em))

    model.train(mode)
    return las, metrics

class Params(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._basic_config()

    def _basic_config(self):
        os.makedirs(self.model_basename, exist_ok=True)

        log = logging.getLogger('dl4dp.training')
        log.setLevel(logging.INFO)
        log.addHandler(FileHandler(self.model_basename + 'training.log', mode='w'))
        self.logger = log

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    preprocess = pipe().only_words().upos_feats().\
        lowercase('form').\
        replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')

    params = Params(max_epoch=2, batch_size=100, model_basename='build/en_ewt/')

    train_data = 'build/en_ewt-ud-train.conllu'
    validation_data = 'build/en_ewt-ud-dev.conllu'
    #test_data = 'build/en_ewt-ud-test.conllu'

    print('building index...')
    index = pipe().read_conllu(train_data).pipe(preprocess).create_index({'form', 'upos_feats', 'deprel'})
    print('building index done')

    params.train_data = pipe().read_conllu(train_data).pipe(preprocess).\
        to_instance(index).collect()
    params.validation_data = pipe().read_conllu(validation_data).pipe(preprocess).\
        to_instance(index).collect()

    embedding_dims = {'form': (len(index['form']) + 1, 100), 'upos_feats': (len(index['upos_feats']) + 1, 100)}
    labels_dim = len(index['deprel']) + 1
    model = BiaffineParser(embedding_dims, labels_dim)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.9))
    train(model, optimizer, params)
