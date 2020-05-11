import os
from pathlib import Path
import logging
from logging import FileHandler

import torch
import numpy as np

from conllutils import pipe
from .tagger import BiaffineTagger
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

def get_model(model_type, *args, **kwargs):
    if model_type == 'tagger':
        return BiaffineTagger(*args, **kwargs)
    if model_type == 'parser':
        return BiaffineParser(*args, **kwargs)

def get_validator(model_type, treebank, *args, **kwargs):
    if treebank is None:
        return None
    if model_type == 'parser':
        return LASValidator(treebank, *args, **kwargs)
    return None

def get_dims(dims, index):

    def _dims(f, c, sub=None):
        n = len(c) + 1
        if isinstance(dims, dict):
            embd = dims[f][sub] if isinstance(dims[f], dict) else dims[f]
            return (n, embd)
        else:
            return n

    index_dims = {}
    if 'feats' in dims:
        index_dims['feats'] = {}
    if 'upos_feats' in dims:
        index_dims['upos_feats'] = {}

    for f, c in index.items():
        if f in dims and f not in index_dims:
            index_dims[f] = _dims(f, c)
        if 'feats' in dims and f.startswith('feats:'):
            index_dims['feats'][f] = _dims('feats', c, f)
        if 'upos_feats' in dims and (f == 'upos' or f.startswith('feats:')):
            index_dims['upos_feats'][f] = _dims('upos_feats', c, f)

    return index_dims

def log_config(model_dir, logger):
    log = logging.getLogger('dl4dp.' + logger)
    log.setLevel(logging.INFO)
    log.addHandler(FileHandler(model_dir / (logger + '.log'), mode='w'))

def main():
    random_seed = 0
    max_epochs = 10
    model_type = 'tagger'

    model_dir = 'build/en_ewt'
    treebanks = {
        'train': 'build/en_ewt-ud-train.conllu',
        'dev': 'build/en_ewt-ud-dev.conllu',
        'test': 'build/en_ewt-ud-test.conllu'
        }

    dims = {'form': 100, 'form:chars': (32, 50), 'upos_feats': 100}

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
    p.split_chars('form')
    p.lowercase('form')
    p.replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')

    index = build_index(treebanks, p)
    treebanks = load_treebanks(treebanks, p, index)

    input_dims = get_dims(dims, index)
    output_dims = get_dims({'upos', 'feats', 'deprel'}, index)

    model = get_model(model_type, input_dims, output_dims)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    validator = get_validator(model_type, treebanks.get('dev'), logger='dl4dp.validation')
    trainer = Trainer(model_dir, max_epochs=max_epochs, validator=validator, logger='dl4dp.training')
    best_path = trainer.train(model, treebanks['train'])

    if 'test' in treebanks:
        model = torch.load(best_path)
        print('testing:')
        get_validator(model_type, treebanks['test']).validate(model)
