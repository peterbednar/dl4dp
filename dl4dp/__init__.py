import os
from pathlib import Path
import logging
from logging import FileHandler

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

def load_treebanks(files, p, index):
    treebanks = {}
    for name, f in files.items():
        data = pipe().read_conllu(f).pipe(p).to_instance(index).collect()
        print(f'{name}: {len(data)} sentences, {sum([i.length for i in data])} words')
        treebanks[name] = data
    return treebanks

def get_dims(dims, index):

    def _dims(f, c):
        n = len(c) + 1
        return (n, dims[f]) if isinstance(dims, dict) else n

    index_dims = {}
    if 'feats' in dims:
        index_dims['feats'] = {}
    if 'upos_feats' in dims:
        index_dims['upos_feats'] = {}

    for f, c in index.items():
        if f in dims:
            index_dims[f] = _dims(f, c)
        if 'feats' in dims and f.startswith('feats:'):
            index_dims['feats'] = _dims('feats', c)
        if 'upos_feats' in dims and (f == 'upos' or f.startswith('feats:')):
            index_dims['upos_feats'][f] = _dims('upos_feats', c)

    return index_dims

def log_config(model_dir, logger):
    log = logging.getLogger('dl4dp.' + logger)
    log.setLevel(logging.INFO)
    log.addHandler(FileHandler(model_dir / (logger + '.log'), mode='w'))

def main():
    random_seed = 0
    max_epoch = 10

    model_dir = 'build/en_ewt'
    treebanks = {
        'train': 'build/en_ewt-ud-train.conllu',
        'dev': 'build/en_ewt-ud-dev.conllu',
        'test': 'build/en_ewt-ud-test.conllu'
        }

    dims = {'form': 100, 'upos_feats': 100}

    model_dir = Path(model_dir)
    log_config(model_dir, 'training')
    log_config(model_dir, 'validation')

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
    treebanks = load_treebanks(treebanks, p, index)

    input_dims = get_dims(dims, index)
    output_dims = get_dims({'deprel'}, index)

    model = BiaffineParser(input_dims, output_dims)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    validator = None
    if 'dev' in treebanks:
        validator = LASValidator(treebanks['dev'], logger='dl4dp.validation')
    trainer = Trainer(model_dir, max_epoch=max_epoch, validator=validator, logger='dl4dp.training')
    best_path = trainer.train(model, treebanks['train'])

    if 'test' in treebanks:
        model = torch.load(best_path)
        print('testing:')
        LASValidator().validate(model, treebanks['test'])
