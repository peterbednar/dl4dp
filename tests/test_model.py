
import numpy as np
from collections import OrderedDict
from dl4dp.model import BiaffineParser, _get_gold_arcs, _get_gold_labels

def test_biaffine_parser():
    instances = [
        {
            "form":np.array([1, 2, 3, 4, 5, 6], dtype=np.int),
            "upos_feats":np.array([1, 1, 2, 2, 3, 3], dtype=np.int),
            "head":np.array([0, 1, 2, 3, 4, 5], dtype=np.int),
            "deprel":np.array([2, 2, 3, 4, 3, 2], dtype=np.int)
        },
        {
            "form":np.array([2, 4, 6], dtype=np.int),
            "upos_feats":np.array([1, 2, 3], dtype=np.int),
            "head":np.array([0, 1, 2], dtype=np.int),
            "deprel":np.array([2, 3, 4], dtype=np.int)
        }
    ]
    
    embedding_dims = OrderedDict([("form", (7, 4)), ("upos_feats", (4, 3))])
    labels_dim = 5

    parser = BiaffineParser(embedding_dims=embedding_dims, labels_dim=labels_dim)
    arc_scores, label_scores, batch_lengths = parser(instances)

    assert arc_scores.shape == (2, 7, 7)
    assert label_scores.shape == (2, 7, 7, 5)

    gold_arcs = _get_gold_arcs(instances)
    gold_deps = _get_gold_labels(instances)

    assert gold_arcs.shape == (2, 6)
    assert gold_deps.shape == (2, 6)
