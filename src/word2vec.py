from __future__ import print_function

from utils import FORM, read_conllu, create_dictionary, create_index, map_to_instances
from gensim.models import Word2Vec

UNKNOWN_TOKEN = unicode("__unknown__")

class Tokens(object):

    def __init__(self, filename, field, index):
        self.filename = filename
        self.field = field
        self.index = index
 
    def __iter__(self):
        for sentence in read_conllu(self.filename):
            tokens = [token[self.field] if self.index[self.field][token[self.field]] != 0 else UNKNOWN_TOKEN for token in sentence]
            yield tokens

if __name__ == "__main__":
    dic = create_dictionary(read_conllu("../treebanks/train/cs/cs.conllu"), fields={FORM})
    index = create_index(dic)

    tokens = Tokens("../treebanks/train/cs/cs.conllu", FORM, index)
    model = Word2Vec(tokens, size=100, window=5, min_count=1, workers=4)
    print(model["nebo"])
