
from abc import ABC, abstractmethod
from .utils import progressbar
from conllutils import HEAD, DEPREL

class Metric(ABC):

    def __init__(self):
        self.total = 0
        self.correct = 0

    @abstractmethod
    def __call__(self, gold, parsed):
        raise NotImplementedError()

    @property
    def value(self):
        return float(self.correct) / self.total

    def __str__(self):
        return f"{self.__class__.__name__}: {self.value:.4f}"

class UAS(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, gold, parsed):
        for n in range(len(gold)):
            if gold[HEAD][n] == parsed[HEAD][n]:
                self.correct += 1
            self.total += 1

class LAS(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, gold, parsed):
        for n in range(len(gold)):
            if gold[HEAD][n] == parsed[HEAD][n] and gold[DEPREL][n] == parsed[DEPREL][n]:
                self.correct += 1
            self.total += 1

class EMS(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, gold, parsed):
        self.total += 1
        for n in range(len(gold)):
            if gold[HEAD][n] != parsed[HEAD][n] or gold[DEPREL][n] != parsed[DEPREL][n]:
                return
        self.correct += 1

def validate(model, validation_data, metrics=[UAS, LAS, EMS]):
    metrics = [metric() for metric in metrics]

    pb = progressbar(len(validation_data))
    for gold in validation_data:
        parsed = model.parse(gold)
        for metric in metrics:
            metric(gold, parsed)
        pb.update(1)
    pb.finish()

    print(", ".join(str(metric) for metric in metrics))
    return [metric.value for metric in metrics]
