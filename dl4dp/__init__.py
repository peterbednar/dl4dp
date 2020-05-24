import copy

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

def pipeline(models, index, batch_size=100):
    inverse_index = create_inverse_index(index)
    p = pipe()
    p = p.map(lambda instance: copy.deepcopy(instance))
    p = p.pipe(preprocess())
    p = p.to_instance(index)
    p = p.batch(batch_size)
    for model in models:
        p = p.map(apply_model(model))
    p.flatten()
    p.to_sentence(inverse_index)
    return p

def apply_model(model):
    def _apply_model(batch):
        preds = model.parse(batch, unbind=True, device='cpu')
        for f, pred in preds.items():
            for instance, value in zip(batch, pred):
                instance[f] = value
    return _apply_model

