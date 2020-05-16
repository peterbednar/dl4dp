import os
import re
import tarfile
from pathlib import Path

import torch
import numpy as np
from conllutils import pipe

from .tagger import BiaffineTagger
from .parser import BiaffineParser
from .trainer import Trainer, UPosFeatsAcc, LAS
from .utils import register_logger, get_url

def preprocess():
    p = pipe()
    p.only_words()
    p.only_fields('form', 'upos', 'feats', 'head', 'deprel')
    p.unwind_feats()
    p.split_chars('form')
    p.lowercase('form')
    p.replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')
    return p

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

def home_dir():
    path = Path('./home')
    path.mkdir(parents=True, exist_ok=True)
    return path

def treebank_dir(treebank=None, create=False):
    path = home_dir() / 'treebanks' if treebank is None else home_dir() / 'treebanks' / treebank
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

def _match_treebank_name(treebank, name):
    return re.compile(fr'.*\/{treebank}-ud-(train|test|dev).conllu').match(name)

def get_treebank_files(treebank):
    tb_dir = treebank_dir(treebank)
    if not tb_dir.exists():
        return extract_ud_treebank(treebank)
    files = {}
    for name in tb_dir.iterdir():
        match = _match_treebank_name(treebank, str(name))
        if match:
            files[match.group(1)] = name
    return files

_UD_FILE = 'ud-treebanks-v2.6.tgz'
_UD_URL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz?sequence=1&isAllowed=y'

def extract_ud_treebank(treebank):
    td_dir = treebank_dir()
    archive = td_dir / _UD_FILE

    if not archive.exists():
        td_dir.mkdir(exist_ok=True)
        print('downloading ' + _UD_FILE, flush=True)
        get_url(_UD_URL, archive)
    
    files = {}
    print(f'extracting {treebank} from UD archive...')

    with tarfile.open(archive, 'r', encoding='utf-8') as tar:
        for member in tar.getmembers():
            match = _match_treebank_name(treebank, member.name)
            if match:
                member.name = Path(member.name).name    # Extract only file name without path
                td_dir = treebank_dir(treebank, create=True)
                tar.extract(member, td_dir)
                files[match.group(1)] = td_dir / member.name

    print('extracting done')
    if not files:
        raise ValueError(f'Unknown treebank {treebank}.')

    return files

def get_model(model_type, *args, **kwargs):
    if model_type == 'tagger':
        return BiaffineTagger(*args, **kwargs)
    if model_type == 'parser':
        return BiaffineParser(*args, **kwargs)

def get_validator(model_type, treebank, *args, **kwargs):
    if treebank is None:
        return None
    if model_type == 'tagger':
        return UPosFeatsAcc(treebank, *args, **kwargs)
    if model_type == 'parser':
        return LAS(treebank, *args, **kwargs)
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

def train(config):
    treebank    = config['treebank']
    random_seed = config.get('random_seed', 0)
    max_epochs  = config.get('max_epochs', 10)
    model_type  = config.get('model_type', 'parser')
    model_dir   = config.get('model_dir', home_dir() / f'build/{treebank}')
    files       = config.get('files', get_treebank_files(treebank))
    embedding_dims = config.get('embedding_dims', {'form': 100, 'form:chars': (32, 50), 'upos_feats': 100})

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    register_logger('training', model_dir / 'training.csv')
    register_logger('validation', model_dir / 'validation.csv')

    p = preprocess()
    index = build_index(files, p)
    treebanks = load_treebanks(files, p, index)

    input_dims = get_dims(embedding_dims, index)
    output_dims = get_dims({'upos', 'feats', 'deprel'}, index)

    model = get_model(model_type, input_dims, output_dims)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    validator = get_validator(model_type, treebanks.get('dev'), logger='validation')
    trainer = Trainer(
        max_epochs=max_epochs,
        logger='training',
        model_dir=model_dir,
        model_name=model_type,
        validator=validator
    )

    print('training ' + model_type)
    best, _ = trainer.train(model, treebanks['train'])

    test = get_validator(model_type, treebanks.get('test'))
    if test:
        model = torch.load(best.path)
        print('testing:')
        test(model)

def main():
    config = {'treebank': 'en_ewt', 'max_epochs': 1}
    train(config)
