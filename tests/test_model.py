
import numpy as np
from collections import OrderedDict
from dl4dp.model import BiaffineParser

def test_biaffine_parser():
    instances = [
        {"a":np.array([1, 2, 3, 4, 5, 6], dtype=np.int), "b":np.array([1, 1, 2, 2, 3, 3], np.int)},
        {"a":np.array([2, 4, 6], dtype=np.int), "b":np.array([1, 2, 3], np.int)}
    ]
    
    embedding_dims = OrderedDict([("a", (7, 4)), ("b", (4, 3))])
    labels_dim = 5

    parser = BiaffineParser(embedding_dims=embedding_dims, labels_dim=labels_dim)
    arc_scores, label_scores = parser(instances)

    assert arc_scores.shape == (2, 7, 7)
    assert label_scores.shape == (2, 7, 7, 5)
