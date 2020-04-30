import time
from datetime import timedelta
import logging
from logging import FileHandler

import torch
from torch.optim import Adam
from conllutils import pipe
from .utils import progressbar

class Trainer(object):

    def __init__(self, model_basename=None, max_epoch=1, batch_size=100, validator=None, logger=None):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.model_basename = model_basename
        self.validator = validator
        if logger:
            self.logger = logger
        else:
            log = logging.getLogger('dl4dp.training')
            log.setLevel(logging.INFO)
            log.addHandler(FileHandler(self.model_basename + 'training.log', mode='w'))
            self.logger = log

    def train(self, model, train_data):
        best_epoch = 0
        best_score = None

        total_size = len(train_data)
        if total_size % self.batch_size:
            total_size += self.batch_size - (total_size % self.batch_size)

        pb = progressbar(total_size)
        optimizer = self._optimizer(model)

        for epoch in range(self.max_epoch):
            print(f'epoch: {epoch + 1}/{self.max_epoch}')
            start_time = time.time()

            for step, batch in enumerate(pipe(train_data).stream(total_size).shuffle().batch(self.batch_size)):
                optimizer.zero_grad()
                loss, metrics = model.loss(batch)
                loss.backward()
                optimizer.step()

                pb.update(self.batch_size)
                elapsed_time = time.time() - start_time
                num_words = sum([instance.length for instance in batch])

                if self.logger:
                    self.logger.info(f'{epoch + 1} {step + 1} {timedelta(seconds=elapsed_time)} {num_words} {loss.item()} ' +
                        ' '.join([str(metric.item()) for metric in metrics]))

            torch.save(model, self.model_basename + f'model-{epoch + 1}.pth')
            pb.finish()
            pb.reset()

            if self.validator:
                print(f'validating epoch {epoch + 1}/{self.max_epoch}')
                score, metrics = self.validator.validate(model)
                if best_score is None or best_score < score:
                    best_score = score
                    best_epoch = epoch + 1
            else:
                best_epoch = epoch

        if best_score is not None:
            print(f'best epoch: {best_epoch}, score: {best_score:.4f}')
        return best_epoch, best_score

    def _optimizer(self, model):
        return Adam(model.parameters(), betas=(0.9, 0.9))

class LASValidator(object):
    
    def __init__(self, validation_data, batch_size=100):
        self.validation_data = validation_data
        self.batch_size = batch_size

    def validate(self, model):
        mode = model.training
        model.eval()

        total_size = len(self.validation_data)
        pb = progressbar(total_size)
        uas_correct = las_correct = em_correct = total = 0

        for batch in pipe(self.validation_data).batch(self.batch_size):
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
        print(', '.join(f"{metric[0]}:{metric[1]:.4f}" for metric in metrics))

        model.train(mode)
        return las, metrics
