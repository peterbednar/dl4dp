from __future__ import print_function

import numpy as np

from utils import FORM, XPOS
from utils import create_dictionary, create_index, create_inverse_index, read_conllu, map_to_instances
from utils import parse_nonprojective, parse_projective, is_projective

FIELDS = (FORM, XPOS)

TRAIN_FILE = "../treebanks/train/en/en.conllu"

def load_data(filename=TRAIN_FILE):
    index = create_index(create_dictionary(read_conllu(TRAIN_FILE, normalize=None)), min_frequency=1)
    train_data = list(map_to_instances(read_conllu(TRAIN_FILE, normalize=None), index, FIELDS))
    return index, train_data

def to_string(tree, inverse_index):
    return [[inverse_index[FIELDS[f]][tree.feats[i,f]] if tree.feats[i,f] != 0 else None for f in range(len(FIELDS))] for i in range(len(tree))]

def test_index():
    index, train_data = load_data()
    inverse_index = create_inverse_index(index)
    sentences = list(read_conllu(TRAIN_FILE, normalize=None))

    for tree, sentence in zip(train_data, sentences):
        assert len(tree) == len(sentence)
        tree_str = to_string(tree, inverse_index)
        for i in range(len(tree)):
            for f in range(len(FIELDS)):
                assert tree_str[i][f] == sentence[i][FIELDS[f]]

def test_parse_nonprojective():
    index, train_data = load_data()

    for tree in train_data:
        num_nodes = len(tree) + 1
        scores = np.random.rand(num_nodes, num_nodes)
        for i in range(1,num_nodes):
            cmax = max(scores[j][i] for j in range(num_nodes))
            gold = tree.heads[i-1]
            scores[gold][i] = cmax + 1e-5

        heads = parse_nonprojective(scores)
        for i in range(len(tree)):
            assert heads[i] == tree.heads[i]

if __name__ == "__main__":
    test_index()
    test_parse_nonprojective()
    test_parse_projective()
    print("all tests passed")
