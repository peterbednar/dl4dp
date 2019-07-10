import numpy as np
from argparse import ArgumentParser
from gensim.models import Word2Vec

from .utils import str_to_field, field_to_str
from .utils import read_conllu, create_dictionary, create_index, write_index, open_file

UNKNOWN_TOKEN = u"__unknown__"
NONE_TOKEN = u"__none__"

def read_word2vec(file, skip_shape=True):

    def _tokenize(l):
        return l.rstrip(" \r\n").split(" ")

    if isinstance(file, str):
        file = open_file(file)
    with file:
        l = file.readline()
        if not skip_shape:
            tokens = _tokenize(l)
            yield (int(tokens[0]), int(tokens[1]))
        for l in file:
            tokens = _tokenize(l)
            w = tokens[0]
            if w == NONE_TOKEN:
                w = None
            v = np.array([float(t) for t in tokens[1:]], dtype=np.float)
            yield (w, v)

def index_word2vec(vectors, index):
    for w, v in vectors:
        i = 0 if w == UNKNOWN_TOKEN else index.get(w)
        if i is not None:
            yield (i, v)

VECTORS_FILENAME = "{0}vectors_{1}.txt"

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
            print(f"building {field_to_str(f).upper()}[{args.size[i]}] vectors...")
            tokens = _Tokens(args.inputfile, f, index)
            model = Word2Vec(tokens, sg=1 if args.sg else 0, size=args.size[i], window=args.window, min_count=1, workers=4, seed=args.seed)
            model.wv.save_word2vec_format(VECTORS_FILENAME.format(args.outbasename, field_to_str(f)))
            print("done")

def _parse_args():
    parser = ArgumentParser()

    parser.add_argument("--inputfile", required=True)
    parser.add_argument("--outbasename", default="")
    parser.add_argument("--fields", default=("FORM", "XPOS"), nargs='+')
    parser.add_argument("--size", default=(100, 100), type=int, nargs='+')
    parser.add_argument("--min_frequency", default=5, type=int)
    parser.add_argument("--window", default=5, type=int)
    parser.add_argument("--sg")
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()
    args.fields = [str_to_field(f) for f in args.fields]
    return args

if __name__ == "__main__":
    args = _parse_args()

    print("building index...")
    dic = create_dictionary(read_conllu(args.inputfile), fields=args.fields)
    index = create_index(dic, min_frequency=args.min_frequency)
    print("building index done")
    write_index(index, args.fields, args.outbasename)

    _word2vec(index, args)
