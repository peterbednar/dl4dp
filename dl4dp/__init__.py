import random
from torch.optim import Adam

from conllutils import FORM_NORM, UPOS_FEATS, DEPREL
from conllutils import read_conllu, create_dictionary, create_index, map_to_instances, shuffled_stream
from .model import BiaffineParser

def train(model, optimizer, params):

    step = 0
    for epoch in range(params.max_epoch):
        print(f"epoch: {epoch + 1}")

        for batch in shuffled_stream(params.train_data, total_size=len(params.train_data), batch_size=params.batch_size):
            optimizer.zero_grad()
            arc_loss, label_loss = model.loss(batch)

            loss = arc_loss # + label_loss
            loss.backward()
            optimizer.step()

            step += 1
            print(f"{epoch + 1} {step} {arc_loss.item()}")

class Params(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def main():
    random.seed(1)
    params = Params({"max_epoch": 1, "batch_size": 100})
    train_data = "build/en_ewt-ud-train.conllu"

    print("building index...")
    index = create_index(create_dictionary(read_conllu(train_data), fields={FORM_NORM, UPOS_FEATS, DEPREL}))
    print("building index done")
    params.train_data = list(map_to_instances(read_conllu(train_data), index))

    embedding_dims = {FORM_NORM: (len(index[FORM_NORM]) + 1, 100), UPOS_FEATS: (len(index[UPOS_FEATS]) + 1, 100)}
    labels_dim = len(index[DEPREL]) + 1
    model = BiaffineParser(embedding_dims, labels_dim)

    optimizer = Adam(model.parameters())
    train(model, optimizer, params)
