import os
from pathlib import Path

import torch
import numpy as np

from conllutils import pipe
from .parser import BiaffineParser
from .trainer import Trainer, LASValidator

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model_path = 'build/en_ewt'
    train_data = 'build/en_ewt-ud-train.conllu'
    validation_data = 'build/en_ewt-ud-dev.conllu'
    test_data = 'build/en_ewt-ud-test.conllu'

    p = pipe()
    p.only_words()
    p.upos_feats()
    p.lowercase('form')
    p.replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')

    print('building index...')
    index = pipe().read_conllu(train_data).pipe(p).create_index({'form', 'upos_feats', 'deprel'})
    print('building index done')

    train_data = pipe().read_conllu(train_data).pipe(p).to_instance(index).collect()
    validation_data = pipe().read_conllu(validation_data).pipe(p).to_instance(index).collect()
    test_data = pipe().read_conllu(test_data).pipe(p).to_instance(index).collect()

    embedding_dims = {'form': (len(index['form']) + 1, 100), 'upos_feats': (len(index['upos_feats']) + 1, 100)}
    labels_dim = len(index['deprel']) + 1
    model = BiaffineParser(embedding_dims, labels_dim)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    trainer = Trainer(model_path, max_epoch=1, validator=LASValidator(validation_data))
    _, _, best_path = trainer.train(model, train_data)

    model = torch.load(best_path)
    LASValidator().validate(model, test_data)
