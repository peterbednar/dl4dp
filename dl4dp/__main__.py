import re
import tarfile
import argparse
from pathlib import Path

import torch
import yaml
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

def load_treebank(files, p, index):
    treebank = {}
    for name, f in files.items():
        data = pipe().read_conllu(f).pipe(p).to_instance(index).collect()
        print(f'{name}: {len(data)} sentences, {sum([i.length for i in data])} words')
        treebank[name] = data
    return treebank

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

_home_dir = Path('./home')

def home_dir():
    path = _home_dir
    path.mkdir(parents=True, exist_ok=True)
    return path

def set_home_dir(path):
    _home_dir = Path(path)

def _get_dir(dir, treebank=None, create=False):
    path = home_dir() / dir
    if treebank is not None:
        path /= treebank
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

def get_treebank_dir(treebank=None, create=False):
    return _get_dir('treebanks', treebank, create)

def get_build_dir(treebank, create=False):
    return _get_dir('build', treebank, create)

def get_model_dir(create=False):
    return _get_dir('models', None, create)

def get_model_name(treebank, version=None):
    name = 'model' if treebank is None else treebank
    if version is not None:
        name += '-' + (version if isinstance(version, str) else '.'.join(str(v) for v in version))
    return name + '.tar.gz'
   
_FILE_NAME = re.compile(r'.*-(train|test|dev).conllu')

def get_treebank_files(treebank, extract_ud=True):
    tb_dir = get_treebank_dir(treebank)

    if not tb_dir.exists() and extract_ud:
        return extract_ud_treebank(treebank)

    files = {}
    for name in tb_dir.iterdir():
        match = _FILE_NAME.match(str(name))
        if match:
            files[match.group(1)] = name

    if not files:
        raise ConfigError(f'no data found in treebank directory {tb_dir}')

    return files

_UD_FILE = 'ud-treebanks-v2.6.tgz'
_UD_URL = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3226/ud-treebanks-v2.6.tgz?sequence=1&isAllowed=y'

def _match_ud_treebank_name(treebank, name):
    return re.compile(fr'.*\/{treebank}-ud-(train|test|dev).conllu').match(name)

def extract_ud_treebank(treebank):
    td_dir = get_treebank_dir(create=True)
    archive = td_dir / _UD_FILE

    if not archive.exists():
        print('downloading ' + _UD_FILE, flush=True)
        get_url(_UD_URL, archive)
    
    files = {}
    print(f'extracting {treebank} from UD archive...')

    with tarfile.open(archive, 'r', encoding='utf-8') as tar:
        for member in tar.getmembers():
            match = _match_ud_treebank_name(treebank, member.name)
            if match:
                member.name = Path(member.name).name  # Extract only file name without path
                td_dir = get_treebank_dir(treebank, create=True)
                tar.extract(member, td_dir)
                files[match.group(1)] = td_dir / member.name

    if not files:
        raise ConfigError(f'treebank {treebank} not found in the UD archive')
    print('extracting done')

    return files

class ConfigError(Exception):

    def __init__(self, message):
        self.message = message

def get_config(args):
    config = {
        'model_type': args.model_type
    }

    if args.config is not None:
        with open(args.config, 'rt', encoding='utf-8') as f:
            config.update(yaml.safe_load(f))

    if args.treebank is not None:
        config['treebank'] = args.treebank

    return config

def train(model_type,
          treebank=None,
          random_seed=0,
          enable_gpu=True,
          **kwargs):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    build_dir = get_build_dir(treebank)
    files = get_treebank_files(treebank)

    register_logger('training', build_dir / f'{model_type}-training.csv')
    register_logger('validation', build_dir / f'{model_type}-validation.csv')

    p = preprocess()
    index = build_index(files, p)
    data = load_treebank(files, p, index)

    build_dir.mkdir(parents=True, exist_ok=True)
    torch.save(index, build_dir / 'index.pth')

    model_config = kwargs.get(model_type, {})
    embedding_dims = model_config.get('embedding_dims', {'form': 100, 'form:chars': (32, 50), 'upos_feats': 100})

    input_dims = get_dims(embedding_dims, index)
    output_dims = get_dims({'upos', 'feats', 'deprel'}, index)

    model = get_model(model_type, input_dims, output_dims, **model_config)
    if enable_gpu and torch.cuda.is_available():
        model.to(torch.device('cuda'))

    validator = get_validator(model_type, data.get('dev'), logger='validation')
    trainer_config = kwargs.get('trainer', {})
    trainer = Trainer(
        data['train'],
        build_dir=build_dir,
        model_name=model_type,
        validator=validator,
        logger='training',
        **trainer_config
    )

    print(f'training {model_type if treebank is None else treebank + " " + model_type}')
    best, _ = trainer.train(model)

    test = get_validator(model_type, data.get('test'))
    if test:
        model = torch.load(best.path)
        print('testing:')
        test(model)

_DEFAULT_VERSION = (0,1,0)

def create_model_package(treebank=None, files=None, version=None, update=False):
    build_dir = get_build_dir(treebank)
    model_dir = get_model_dir(create=True)

    if files is None:
        files = {'*.pth', 'LICENSE'}

    if version is None:
        version = _DEFAULT_VERSION

    package_files = set()
    for p in files:
        package_files |= set(build_dir.glob(p))
    if not package_files:
        raise ConfigError(f'no package files found in {build_dir}')

    path = model_dir / get_model_name(treebank, version)
    print(f'creating package {path.name} ...')
    if path.exists() and not update:
        raise ConfigError(f'package {path.name} is already installed, use --update option to overwrite')

    with tarfile.open(path, 'w:gz') as tar:
        for f in package_files:
            tar.add(f, arcname=f.name)
    print('updating done' if update else 'installation done')

def get_argparser():
    p = argparse.ArgumentParser(prog='dl4dp')
    p.add_argument('--home', dest='home_dir')
    ps = p.add_subparsers(dest='cmd')
    ps.required = True

    train = ps.add_parser('train')
    train.add_argument('model_type', choices=('tagger','parser'))
    train.add_argument('-t', '--treebank')
    train.add_argument('-c', '--config')

    parse = ps.add_parser('parse')
    parse.add_argument('input')
    parse.add_argument('output')
    parse.add_argument('-m', '--model')

    package = ps.add_parser('package')
    package.add_argument('package_opr', choices=('install','update'))
    package.add_argument('-t', '--treebank')
    package.add_argument('-v', '--version')
    return p

def main():
    args = get_argparser().parse_args()

    if args.home_dir is not None:
        set_home_dir(args.home_dir)

    try:
        if args.cmd == 'train':
            train(**get_config(args))
        elif args.cmd == 'package':
            opr = args.package_opr
            if opr == 'install' or opr == 'update':
                create_model_package(args.treebank, update=opr=='update', version=args.version)
    except ConfigError as err:
        print('error:', err.message)

if __name__ == "__main__":
    main()
