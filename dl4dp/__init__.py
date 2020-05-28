import re
import copy
import tarfile

import torch
from conllutils import pipe, create_inverse_index

def preprocess():
    p = pipe()
    p.only_words()
    p.only_fields('form', 'upos', 'feats', 'head', 'deprel')
    p.unwind_feats()
    p.split_chars('form')
    p.lowercase('form')
    p.replace('form', r'[0-9]+|[0-9]+\.[0-9]+|[0-9]+[0-9,]+', '__number__')
    return p

def apply_model(batch, model, fields):
    preds = model.parse(batch, unbind=True, device='cpu')
    fields |= preds.keys()
    for f, pred in preds.items():
        for instance, value in zip(batch, pred):
            instance[f] = value
    return batch

def batch_pipeline(models, index):
    inverse_index = create_inverse_index(index)

    p = pipe()
    p = p.map(lambda instance: copy.deepcopy(instance))
    p = p.pipe(preprocess())
    p = p.to_instance(index)

    def _batch_pipeline(batch):
        parsed_fields = set()

        parsed = pipe(batch).pipe(p).collect()
        for model in models:
            parsed = apply_model(parsed, model, parsed_fields)
        parsed = pipe(parsed).only_fields(parsed_fields).map_to_sentence(inverse_index).collect()

        for sentence, parsed_sentence in zip(batch, parsed):
            for i, parsed_word in enumerate(parsed_sentence):
                word = sentence.get(i + 1)
                word.update(parsed_word)

    return _batch_pipeline

_PIPELINE_NAME = re.compile(r'.*(tagger|parser|index).pth')

def load_pipeline(name):
    pipeline = {}
    with tarfile.open(name, 'r:gz') as tar:
        for member in tar.getmembers():
            match = _PIPELINE_NAME.match(member.name)
            if match:
                pipeline[match.group(1)] = torch.load(tar.extractfile(member))
    return pipeline
