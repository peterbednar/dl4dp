import torch
import numpy as np

from collections import OrderedDict
from dl4dp.modules import Embeddings, LSTM

def test_embeddings():
    instances = [
        {"a":np.array([1, 2, 3, 4, 5, 6], dtype=np.int), "b":np.array([1, 1, 2, 2, 3, 3], np.int)},
        {"a":np.array([2, 4, 6], dtype=np.int), "b":np.array([1, 2, 3], np.int)}
    ]

    embeddings = Embeddings(OrderedDict([("a", (7, 4)), ("b", (4, 3))]))
    batch = embeddings(instances)

    assert batch[0].shape == (6, 7)
    assert batch[1].shape == (3, 7)

def test_lstm():
    instances = [
        {"a":np.array([1, 2, 3, 4, 5, 6], dtype=np.int), "b":np.array([1, 1, 2, 2, 3, 3], np.int)},
        {"a":np.array([2, 4, 6], dtype=np.int), "b":np.array([1, 2, 3], np.int)}
    ]

    embeddings = Embeddings(OrderedDict([("a", (7, 4)), ("b", (4, 3))]))
    lstm = LSTM(7, 5, 1)
    h = lstm(embeddings(instances))

    assert h.shape == (2, 7, 5 * 2)
