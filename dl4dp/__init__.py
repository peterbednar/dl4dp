import os
from pathlib import Path

import torch
import numpy as np

from conllutils import pipe
from .parser import BiaffineParser
from .trainer import Trainer, LASValidator

def build_index(treebanks, p):
    print('building index...')
    index = pipe().read_conllu(treebanks['train']).pipe(p).create_index(missing_index=1)
    print('building index done')
    return index

def load_data(files, p, index):
    treebanks = {}
    for name, f in files.items():
        data = pipe().read_conllu(f).pipe(p).to_instance(index).collect()
        print(f'{name}: {len(data)} sentences, {sum([i.length for i in data])} tokens')
        treebanks[name] = data
    return treebanks

def get_embedding_dims(dims, index):
    emb_dims = {}
    emb_dims['upos_feats'] = {}
    for f, c in index.items():
        if (f == 'upos' or f.startswith('feats:')) and 'upos_feats' in dims:
            emb_dims['upos_feats'][f] = (len(c) + 1, dims['upos_feats'])
        elif f in dims:
            emb_dims[f] = (len(c) + 1, dims[f])
    return emb_dims

def main():
    random_seed = 0
    max_epoch = 10

    model_path = 'build/en_ewt'
    treebanks = {
        'train': 'build/en_ewt-ud-train.conllu',
        'dev': 'build/en_ewt-ud-dev.conllu',
        'test': 'build/en_ewt-ud-test.conllu'
        }

    dims = {'form': 100, 'upos_feats': 100}

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    p = pipe()
    p.only_words()
    p.only_fields('form', 'upos', 'feats', 'head', 'deprel')
    p.unwind_feats()
    p.lowercase('form')
    p.replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')

    index = build_index(treebanks, p)
    treebanks = load_data(treebanks, p, index)

    embedding_dims = get_embedding_dims(dims, index)
    labels_dim = len(index['deprel']) + 1

    model = BiaffineParser(embedding_dims, labels_dim)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    validator = LASValidator(treebanks['dev']) if 'dev' in treebanks else None
    trainer = Trainer(model_path, max_epoch=max_epoch, validator=validator)
    _, _, best_path = trainer.train(model, treebanks['train'])

    model = torch.load(best_path)
    if 'test' in treebanks:
        print('testing:')
        LASValidator().validate(model, treebanks['test'])
