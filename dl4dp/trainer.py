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


class SessionListerner(ABC):

    def begin_session(self, session):
        pass

    def end_session(self, session):
        pass

    def continue_session(self, session):
        return True

    def begin_epoch(self, session):
        pass

    def end_epoch(self, session):
        pass

    def begin_step(self, session, batch):
        pass

    def end_step(self, session, batch, loss, metrics):
        pass


class SessionLogger(SessionListerner):

    def __init__(self, logger=None):
        if isinstance(logger, str):
            logger = get_logger(logger)
        self.logger = logger

    def begin_session(self, session):
        self.start_time = time.time()
        self.epoch = self.steps = 0
        self.progress = progressbar(session.total_size)

    def end_session(self, session):
        td = timedelta(seconds=round(time.time() - self.start_time))
        print(f'training epochs: {self.epoch}, steps: {self.steps}/{session.batch_size}, elapsed time: {td}')

    def begin_epoch(self, session):
        self.epoch += 1
        self.step = 0
        self.progress.reset()
        print(
            f'epoch: {self.epoch}' if session.max_epochs is None else
            f'epoch: {self.epoch}/{session.max_epochs}'
        )

    def end_epoch(self, session):
        self.progress.finish()
        self.progress.print_elapsed_time('sentences')

    def begin_step(self, session, batch):
        self.step += 1
        self.steps += 1

    def end_step(self, session, batch, loss, metrics):
        self.progress.update(len(batch))
        if self.logger:
            record = {
                'epoch': self.epoch,
                'step': self.step,
                'elapsed_time': self.progress.elapsed_time().total_seconds(),
                'batch_sentences': len(batch),
                'batch_words': sum((instance.length for instance in batch)),
                'loss': loss.item()
            }
            record.update({f: m.item() for f, m in metrics.items()})
            self.logger.log(record)


class Checkpoint(object):

    def __init__(self, epoch, score=None, path=None):
        self.epoch = epoch
        self.score = score
        self.path = path


class CheckpointManager(SessionListerner):

    def __init__(self, build_dir, model_name, validator=None, check_best_only=True, max_epochs=1, patience=None, min_delta=0.001):
        if isinstance(build_dir, str):
            build_dir = Path(build_dir)
        self.build_dir = build_dir
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.validator = validator
        self.check_best_only = check_best_only
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta

    def begin_session(self, session):
        self.epoch = 0
        self.best = None
        self.history = []
        self.no_improvement = 0

    def end_session(self, session):
        if self.best.score is not None:
            print(f'best epoch: {self.best.epoch}, score: {self.best.score:.4f}')

    def continue_session(self, session):
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            return False
        if self.patience is not None and self.no_improvement > self.patience:
            return False
        return True

    def begin_epoch(self, session):
        self.epoch += 1

    def end_epoch(self, session):
        last = Checkpoint(self.epoch)
        self.history.append(last)

        if self.validator:
            print(f'validating epoch: {self.epoch}')
            score, _ = self.validator(session.model)
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
            last.path = self.build_dir / (self.model_name + '.pth' if self.check_best_only else f'_{self.epoch}.pth')
            torch.save(session.model, last.path)

    def get_best(self):
        return self.best, self.history


class Trainer(object):

    def __init__(self, data, steps_per_epoch=None, max_epochs=1, batch_size=100, logger=None, **kwargs):
        if steps_per_epoch is None:
            steps_per_epoch = math.ceil(len(data) / batch_size)

        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.total_size = self.steps_per_epoch * self.batch_size

        self.data = data
        self.batches = pipe(data).stream(self.total_size).shuffle().batch(self.batch_size)

        self.checkpoints = CheckpointManager(max_epochs=max_epochs, **kwargs)
        self.listeners = [
            SessionLogger(logger),
            self.checkpoints
        ]

    def train(self, model):
        optimizer = self._optimizer(model)
        self.model = model
        self._event("begin_session", self)

        while True:
            self._event("begin_epoch", self)

            for batch in self.batches:
                self._event("begin_step", self, batch)
                optimizer.zero_grad()
                loss, metrics = model.loss(batch)
                loss.backward()
                optimizer.step()
                self._event("end_step", self, batch, loss, metrics)

            self._event("end_epoch", self)
            if not self._continue_session():
                break

        self._event("end_session", self)
        return self.checkpoints.get_best()

    def _optimizer(self, model):
        return Adam(model.parameters(), betas=(0.9, 0.9))

    def _event(self, event, *args):
        for listener in self.listeners:
            getattr(listener, event)(*args)

    def _continue_session(self):
        for listener in self.listeners:
            if not listener.continue_session(self):
                return False
        return True


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
