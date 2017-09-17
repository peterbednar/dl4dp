from __future__ import print_function

import codecs
import numpy as np
from argparse import ArgumentParser
from utils import FORM, UPOS, FEATS, FIELD_TO_STR, STR_TO_FIELD
from utils import read_conllu, create_dictionary, create_index, write_index, read_index

UNKNOWN_TOKEN = u"__unknown__"
NONE_TOKEN = u"__none__"

VECTORS_FILENAME = "{0}_{1}_vectors.txt"

class _Tokens(object):

    def __init__(self, filename, field, index):
        self.filename = filename
        self.field = field
        self.index = index
 
    def __iter__(self):

        def map_token(token):
            token = token[self.field]
            if self.index[self.field][token] == 0:
                token = UNKNOWN_TOKEN
            elif token is None:
                token = NONE_TOKEN
            return token

        for sentence in read_conllu(self.filename):
            tokens = map(map_token, sentence)
            yield list(tokens)

def _word2vec(index, args):
    for i, f in enumerate(args.fields):
        print("building {0} vectors...".format(FIELD_TO_STR[f].upper()), end=" ")
        tokens = _Tokens(args.inputfile, f, index)
        model = Word2Vec(tokens, sg=1 if args.sg else 0, size=args.size[i], window=args.window, min_count=1, workers=4, seed=args.seed)
        model.wv.save_word2vec_format(VECTORS_FILENAME.format(args.outbasename, FIELD_TO_STR[f]))
        print("done")

def read_word2vec(basename, fields=(FORM, UPOS, FEATS), index=None):
    if index is None:
        index = read_index(basename, fields)
    
    vectors = []
    for f in fields:
        with codecs.open(VECTORS_FILENAME.format(basename, FIELD_TO_STR[f]), "r", "utf-8") as fp:
            num_tokens, size = (int(num) for num in fp.readline().split(" "))
            num_tokens = len(index[f]) + 1
            a = np.zeros((num_tokens, size))
            for line in fp:
                line = line.rstrip("\r\n").split(" ")
                token = line[0]
                i = index[f][token]
                if i > 0 or token == UNKNOWN_TOKEN:
                    a[i] = [float(num) for num in line[1:]]
            vectors.append(a)
    
    return vectors

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outbasename", required=True)
    parser.add_argument("--fields", default=["FORM", "UPOS", "FEATS"], nargs='+')
    parser.add_argument("--size", default=[100, 5, 20], type=int, nargs='+')
    parser.add_argument("--min_frequency", default=5, type=int)
    parser.add_argument("--window", default=5, type=int)
    parser.add_argument("--sg")
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()
    args.fields = [STR_TO_FIELD[f.lower()] for f in args.fields]
    return args

if __name__ == "__main__":
    from gensim.models import Word2Vec
    args = _parse_args()

    print("building index...", end=" ")
    dic = create_dictionary(read_conllu(args.inputfile), fields=args.fields)
    index = create_index(dic, min_frequency=args.min_frequency)
    print("done")
    write_index(args.outbasename, index, args.fields)

    _word2vec(index, args)
