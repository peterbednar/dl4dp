import codecs
import numpy as np
from argparse import ArgumentParser
from utils import FORM, XPOS, FIELD_TO_STR, STR_TO_FIELD
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
        if args.size[i] > 0:
            print("building {0}[{1}] vectors...".format(FIELD_TO_STR[f].upper(), args.size[i]), end=" ")
            tokens = _Tokens(args.inputfile, f, index)
            model = Word2Vec(tokens, sg=1 if args.sg else 0, size=args.size[i], window=args.window, min_count=1, workers=4, seed=args.seed)
            model.wv.save_word2vec_format(VECTORS_FILENAME.format(args.outbasename, FIELD_TO_STR[f]))
            print("done")

def _read_word2vec_lines(fp, field, index):
    for line in fp:
        line = line.rstrip("\r\n").split(" ")
        token = line[0]
        i = index[field][token]
        if i > 0 or token == UNKNOWN_TOKEN:
           v = [float(num) for num in line[1:]]
           yield (i,v)

def read_word2vec(basename, fields=(FORM, XPOS), index=None):
    if index is None:
        index = read_index(basename, fields)

    for fi, f in enumerate(fields):
        with codecs.open(VECTORS_FILENAME.format(basename, FIELD_TO_STR[f]), "r", "utf-8") as fp:
            num_tokens, size = (int(num) for num in fp.readline().split(" "))
            for i,vec in _read_word2vec_lines(fp, f, index):
                yield (fi,i,vec)
        
def _parse_args():
    parser = ArgumentParser()

    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outbasename", required=True)
    parser.add_argument("--fields", default=["FORM", "XPOS", "DEPREL"], nargs='+')
    parser.add_argument("--size", default=[100, 25, 0], type=int, nargs='+')
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

    print("building index...")
    dic = create_dictionary(read_conllu(args.inputfile), fields=args.fields)
    index = create_index(dic, min_frequency=args.min_frequency)
    print("building index done")
    write_index(index, args.fields, args.outbasename)

    _word2vec(index, args)
