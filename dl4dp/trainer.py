import time
from datetime import timedelta
import logging
from pathlib import Path

import torch
from torch.optim import Adam
from conllutils import pipe
from .utils import progressbar

class Trainer(object):

    def __init__(self, model_dir=None, max_epoch=1, batch_size=100, validator=None, logger=None):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.validator = validator
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        self.logger = logger

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

                pb.update(len(batch))
                if self.logger:
                    elapsed_time = time.time() - start_time
                    num_words = sum([instance.length for instance in batch])
                    self.logger.info(f'{epoch + 1} {step + 1} {timedelta(seconds=elapsed_time)} {len(batch)} {num_words}'
                        f' {loss.item()} ' + ' '.join([str(metric.item()) for metric in metrics]))

            torch.save(model, self.model_dir / f'model_{epoch + 1}.pth')
            pb.finish()
            pb.reset()

            if self.validator:
                print(f'validating epoch: {epoch + 1}/{self.max_epoch}')
                score, metrics = self.validator.validate(model)
                if best_score is None or best_score < score:
                    best_score = score
                    best_epoch = epoch + 1
            else:
                best_epoch = epoch + 1

        if best_score is not None:
            print(f'best epoch: {best_epoch}, score: {best_score:.4f}')
        return self.model_dir / f'model_{best_epoch}.pth'

    def _optimizer(self, model):
        return Adam(model.parameters(), betas=(0.9, 0.9))

class LASValidator(object):
    
    def __init__(self, validation_data=None, batch_size=100, logger=None):
        self.step = 0
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.logger = logger
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)

    def reset(self):
        self.step = 0

    def validate(self, model, validation_data=None):
        if validation_data is None:
            validation_data = self.validation_data

        mode = model.training
        model.eval()

        total_size = len(validation_data)
        pb = progressbar(total_size)
        uas_correct = las_correct = em_correct = total = 0

        start_time = time.time()
        for batch in pipe(validation_data).batch(self.batch_size):
            pred = model.parse(batch, unbind=True)

            for gold, pred in zip(batch, zip(pred['head'], pred['deprel'])):
                gold_head = torch.from_numpy(gold.head)
                gold_deprel = torch.from_numpy(gold.deprel)

                head_mask = gold_head.eq(pred[0].cpu())
                head_correct = head_mask.sum().item()
                deprel_correct = gold_deprel.eq(pred[1].cpu())[head_mask].sum().item()

                uas_correct += head_correct
                las_correct += deprel_correct
                if deprel_correct == gold.length:
                    em_correct += 1

                total += gold.length
                pb.update()

        pb.finish()
        self.step += 1

        uas = uas_correct / total
        las = las_correct / total
        em = em_correct / total_size
        metrics = (('UAS', uas), ('LAS', las), ('EM', em))
        print(', '.join(f"{metric[0]}:{metric[1]:.4f}" for metric in metrics))

        if self.logger:
            elapsed_time = time.time() - start_time
            self.logger.info(f'{self.step} {timedelta(seconds=elapsed_time)} {total_size} {total} '
                    + ' '.join([str(metric[1]) for metric in metrics]))

        model.train(mode)
        return las, metrics
