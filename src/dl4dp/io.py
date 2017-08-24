from __future__ import print_function

import codecs

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

def read_conllu(filename):

    def parse_token(line):
        columns = line.split("\t")

        if "." in columns[ID]:
            token_id, index = columns[ID].split(".")
            id = (int(token_id), int(index), ".")
        elif "-" in columns[ID]:
            start, end = columns[ID].split("-")
            id = (int(start), int(end), "-")
        else:
            id = int(columns[ID])
        columns[ID] = id

        return columns

    def parse_sentence(lines):
        sentence = [parse_token(line) for line in lines]
        return sentence

    lines = []
    with codecs.open(filename, "r", "utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                if len(lines) != 0:
                    yield parse_sentence(lines)
                break
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                continue
            if not line:
                if len(lines) != 0:
                    yield parse_sentence(lines)
                    lines = []
                continue
            lines.append(line)

if __name__ == "__main__":
    for s in read_conllu("../../test/test1.conllu"):
        print(s)
