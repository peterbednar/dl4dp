from pathlib import Path
from abc import ABC, abstractmethod
from collections import Counter

import torch
from torch.optim import Adam
from conllutils import pipe

from .utils import progressbar, get_logger

class Trainer(object):

    def __init__(self, model_dir=None, max_epochs=1, batch_size=100, validator=None, logger=None):
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validator = validator
        if isinstance(logger, str):
            logger = get_logger(logger)
        self.logger = logger
        self.progress = progressbar()

    def train(self, model, train_data):
        best_epoch = 0
        best_score = None

        total_size = len(train_data)
        if total_size % self.batch_size:
            total_size += self.batch_size - (total_size % self.batch_size)

        self.progress.reset(total=total_size)
        optimizer = self._optimizer(model)

        for epoch in range(self.max_epochs):
            print(f'epoch: {epoch + 1}/{self.max_epochs}')
            self.progress.reset()

            for step, batch in enumerate(pipe(train_data).stream(total_size).shuffle().batch(self.batch_size)):
                optimizer.zero_grad()
                loss, metrics = model.loss(batch)
                loss.backward()
                optimizer.step()

                self.progress.update(len(batch))
                self._log(epoch + 1, step + 1, batch, loss, metrics)
            
            self.progress.finish()
            self.progress.print_elapsed_time('sentences')

            torch.save(model, self.model_dir / f'model_{epoch + 1}.pth')

            if self.validator:
                print(f'validating epoch: {epoch + 1}/{self.max_epochs}')
                score, metrics = self.validator(model)
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
        self.progress = progressbar(len(data))
        self.data = data
        self.batches = pipe(data).batch(batch_size)
        if isinstance(logger, str):
            logger = get_logger(logger)
        self.logger = logger

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
            pred = model.parse(batch, unbind=False)
            total_words += sum([instance.length for instance in batch])
            for f, pr in pred.items():
                gold = self._get_gold(f, batch)
                counts[f] += gold.eq(pr.cpu()).sum().item()
            self.progress.update(len(batch))

        for f in counts:
            counts[f] /= total_words
        upos_acc = counts.pop('upos')
        feats_acc = sum(counts.values()) / len(counts)

        score = (upos_acc + feats_acc)/2 
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
            pred = model.parse(batch, unbind=True)

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
        em  = em_correct  / total_sentences

        metrics = {'LAS': las, 'UAS': uas, 'EM': em}
        return las, metrics
