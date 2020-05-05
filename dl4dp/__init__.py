import os
from pathlib import Path

import torch
import numpy as np

from conllutils import pipe
from .parser import BiaffineParser
from .trainer import Trainer, LASValidator

def build_index(treebanks, p):
    print('building index...')
    index = pipe().read_conllu(treebanks['train']).pipe(p).create_index({'form', 'upos_feats', 'deprel'})
    print('building index done')
    return index

def load_data(files, p, index):
    treebanks = {}
    for name, f in files.items():
        data = pipe().read_conllu(f).pipe(p).to_instance(index).collect()
        print(f'{name}: {len(data)} sentences, {sum([i.length for i in data])} tokens')
        treebanks[name] = data
    return treebanks

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model_path = 'build/en_ewt'
    treebanks = {
        'train': 'build/en_ewt-ud-train.conllu',
        'dev': 'build/en_ewt-ud-dev.conllu',
        'test': 'build/en_ewt-ud-test.conllu'
        }

    p = pipe()
    p.only_words()
    p.upos_feats()
    p.lowercase('form')
    p.replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')

    index = build_index(treebanks, p)
    treebanks = load_data(treebanks, p, index)

    embedding_dims = {'form': (len(index['form']) + 1, 100), 'upos_feats': (len(index['upos_feats']) + 1, 100)}
    labels_dim = len(index['deprel']) + 1
    model = BiaffineParser(embedding_dims, labels_dim)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    validator = LASValidator(treebanks['dev']) if 'dev' in treebanks else None
    trainer = Trainer(model_path, max_epoch=1, validator=validator)
    _, _, best_path = trainer.train(model, treebanks['train'])

    model = torch.load(best_path)
    if 'test' in treebanks:
        print('testing:')
        LASValidator().validate(model, treebanks['test'])
