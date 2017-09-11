from __future__ import print_function

from argparse import ArgumentParser
from utils import FORM, UPOS, FEATS, field_to_str, str_to_field, read_conllu, create_dictionary, create_index, write_index
from gensim.models import Word2Vec

UNKNOWN_TOKEN = u"__unknown__"
NONE_TOKEN = u"__none__"

VECTORS_FILENAME = "{0}_{1}_vectors.txt"

class Tokens(object):

    def __init__(self, filename, field, index):
        self.filename = filename
        self.field = field
        self.index = index
 
    def __iter__(self):
        for sentence in read_conllu(self.filename):
            tokens = []
            for token in sentence:
                token = token[self.field]
                if self.index[self.field][token] == 0:
                    token = UNKNOWN_TOKEN
                elif token is None:
                    token = NONE_TOKEN
                tokens.append(token)
            yield tokens

def word2vec(index, args):
    for i, f in enumerate(args.fields):
        print("building {0} vectors...".format(field_to_str[f].upper()))
        tokens = Tokens(args.inputfile, f, index)
        model = Word2Vec(tokens, sg=1 if args.sg else 0, size=args.size[i], window=args.window, min_count=1, workers=4, seed=args.seed)
        model.wv.save_word2vec_format(VECTORS_FILENAME.format(args.outbasename, field_to_str[f]))
        print("done")

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--inputfile")
    parser.add_argument("--outbasename")
    parser.add_argument("--fields", default=["FORM", "UPOS", "FEATS"], nargs='+')
    parser.add_argument("--size", default=[100, 5, 20], type=int, nargs='+')
    parser.add_argument("--min_frequency", default=5, type=int)
    parser.add_argument("--window", default=5, type=int)
    parser.add_argument("--sg")
    parser.add_argument("--write_index", default=True)
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()
    args.fields = [str_to_field[f.lower()] for f in args.fields]
    return args

if __name__ == "__main__":
    args = parse_args()

    print("building index...")
    dic = create_dictionary(read_conllu(args.inputfile), fields=args.fields)
    index = create_index(dic, min_frequency=args.min_frequency)
    print("done")
    if args.write_index:
        write_index(args.outbasename, index, args.fields)

    word2vec(index, args)
