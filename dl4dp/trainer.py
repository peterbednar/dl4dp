import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import timedelta
from pathlib import Path

import torch
from torch.optim import Adam
from conllutils import pipe

from .utils import progressbar, get_logger


class Checkpoint(object):

    def __init__(self, epoch, score=None, path=None):
        self.epoch = epoch
        self.score = score
        self.path = path


class CheckpointManager(object):

    def __init__(self, build_dir, model_name, validator=None, check_best_only=True, patience=None, min_delta=0.001, **kwargs):
        if isinstance(build_dir, str):
            build_dir = Path(build_dir)
        self.build_dir = build_dir
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.validator = validator
        self.check_best_only = check_best_only
        self.best = None
        self.history = []
        self.patience = patience
        self.min_delta = min_delta
        self.no_improvement = 0

    def check(self, epoch, model):
        last = Checkpoint(epoch)
        self.history.append(last)

        if self.validator:
            print(f'validating epoch: {epoch}')
            score, _ = self.validator(model)
            last.score = score

            if self.best is None:
                self.best = last
            else:
                if score - self.best.score < self.min_delta:
                    self.no_improvement += 1
                if self.best.score < score:
                    self.best = last
        else:
            self.best = last

        if not self.check_best_only or self.best == last:
            last.path = self.build_dir / (self.model_name + '.pth' if self.check_best_only else f'_{epoch}.pth')
            torch.save(model, last.path)

        if self.patience is not None and self.no_improvement > self.patience:
            return True
        else:
            return False


class Trainer(object):

    def __init__(self, data, steps_per_epoch=None, max_epochs=1, batch_size=100, logger=None, **kwargs):
        if steps_per_epoch is None:
            steps_per_epoch = math.ceil(len(data) / batch_size)
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        total_size = self.steps_per_epoch * self.batch_size
        self.data = data
        self.batches = pipe(data).stream(total_size).shuffle().batch(self.batch_size)

        self.checkpoints = CheckpointManager(**kwargs)
        if isinstance(logger, str):
            logger = get_logger(logger)
        self.logger = logger
        self.progress = progressbar(total_size)

    def train(self, model):
        optimizer = self._optimizer(model)

        epoch = steps = 0
        start_time = time.time()

        while True:
            print(
                f'epoch: {epoch + 1}' if self.max_epochs is None else
                f'epoch: {epoch + 1}/{self.max_epochs}'
            )
            self.progress.reset()

            for step, batch in enumerate(self.batches):
                optimizer.zero_grad()
                loss, metrics = model.loss(batch)
                loss.backward()
                optimizer.step()

                self.progress.update(len(batch))
                self._log(epoch + 1, step + 1, batch, loss, metrics)
                steps += 1

            self.progress.finish()
            self.progress.print_elapsed_time('sentences')
            epoch += 1

            if self.checkpoints.check(epoch, model):
                break
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break

        td = timedelta(seconds=round(time.time() - start_time))
        print(f'training epochs: {epoch}, steps: {steps}/{self.batch_size}, elapsed time: {td}')

        best = self.checkpoints.best
        if best.score is not None:
            print(f'best epoch: {best.epoch}, score: {best.score:.4f}')

        return best, self.checkpoints.history

    def _optimizer(self, model):
        return Adam(model.parameters(), betas=(0.9, 0.9))

    def _log(self, epoch, step, batch, loss, metrics):
        if self.logger:
            record = {
                'epoch': epoch,
                'step': step,
                'elapsed_time': self.progress.elapsed_time().total_seconds(),
                'batch_sentences': len(batch),
                'batch_words': sum((instance.length for instance in batch)),
                'loss': loss.item()
            }
            record.update({f: m.item() for f, m in metrics.items()})
            self.logger.log(record)


class Validator(ABC):

    def __init__(self, data, batch_size=100, logger=None):
        self.step = 0
        self.data = data
        self.batches = pipe(data).batch(batch_size)
        if isinstance(logger, str):
            logger = get_logger(logger)
        self.logger = logger
        self.progress = progressbar(len(data))

    @abstractmethod
    def validate(self, model):
        pass

    def __call__(self, model):
        mode = model.training
        model.eval()

        self.progress.reset()
        score, metrics = self.validate(model)
        self.progress.finish()

        self.step += 1
        self._print_metrics(metrics)
        self._log(metrics)

        model.train(mode)
        return score, metrics

    def _print_metrics(self, metrics):
        self.progress.print_elapsed_time('sentences')
        print(', '.join(f'{n}: {v:.4f}' for n, v in metrics.items()))

    def _log(self, metrics):
        if self.logger:
            record = {
                'step': self.step,
                'elapsed_time': self.progress.elapsed_time().total_seconds(),
                'total_sentences': len(self.data),
                'total_words': sum((instance.length for instance in self.data))
             }
            record.update(metrics)
            self.logger.log(record)


class UPosFeatsAcc(Validator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self, model):
        counts = Counter()
        total_words = 0

        for batch in self.batches:
            pred = model.parse(batch, unbind=False, device='cpu')
            total_words += sum([instance.length for instance in batch])
            for f, pr in pred.items():
                gold = self._get_gold(f, batch)
                counts[f] += gold.eq(pr).sum().item()
            self.progress.update(len(batch))

        for f in counts:
            counts[f] /= total_words
        upos_acc = counts.pop('upos')
        feats_acc = sum(counts.values()) / len(counts)

        score = (upos_acc + feats_acc) / 2
        metrics = {'UPosAcc': upos_acc, 'UFeatsAcc': feats_acc}
        return score, metrics

    def _get_gold(self, field, batch):
        return torch.cat([torch.from_numpy(instance[field]) for instance in batch])


class LAS(Validator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self, model):
        em_correct = 0
        uas_correct = las_correct = 0
        total_words = total_sentences = 0

        for batch in self.batches:
            pred = model.parse(batch, unbind=True, device='cpu')

            for gold, pred in zip(batch, zip(pred['head'], pred['deprel'])):
                gold_head = torch.from_numpy(gold.head)
                gold_deprel = torch.from_numpy(gold.deprel)

                head_mask = gold_head.eq(pred[0])
                head_correct = head_mask.sum().item()
                deprel_correct = gold_deprel.eq(pred[1])[head_mask].sum().item()

                uas_correct += head_correct
                las_correct += deprel_correct
                if deprel_correct == gold.length:
                    em_correct += 1

                total_sentences += 1
                total_words += gold.length
                self.progress.update()

        las = las_correct / total_words
        uas = uas_correct / total_words
        em = em_correct / total_sentences

        metrics = {'LAS': las, 'UAS': uas, 'EM': em}
        return las, metrics
