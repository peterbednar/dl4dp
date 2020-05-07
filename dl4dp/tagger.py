import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from .modules import Embedding, Embeddings, MLP, Biaffine, _field_option

class BiaffineTagger(nn.Module):

    def __init__(self,
                 embedding_dims,
                 label_dims,
                 input_dropout=0.33,
                 char_lstm_layers=2,
                 char_lstm_dropout=0.33):
        super().__init__()

        self.embeddings = Embeddings('cat')
        self.embeddings['form'] = Embedding('form', embedding_dims, input_dropout)
        self.embeddings['form:chars'] = CharLSTMEncoder('form:chars', embedding_dims, input_dropout,
                char_lstm_layers, char_lstm_dropout)

def _loss_and_error(scores, gold, criterion):
    pred = scores.max(1)[1]
    loss = criterion(scores, gold)
    error = 1 - (pred.eq(gold).sum() / float(gold.size()[0]))
    return loss, error

class UposAffine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.affine = nn.Linear(mlp_dim, labels_dim, bias=True)

    def forward(self, h):
        x = self.mlp(h)
        y = self.affine(x)
        return y

    def loss(self, h, upos_gold):
        return _loss_and_error(self(h), upos_gold, self.criterion)

    def parse(self, h):
        return self(h).max(1)[1].cpu().numpy()

class FeatsBiaffine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.biaffine = Biaffine(mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, h, upos):
        x = self.mlp(h)
        y = self.biaffine(x, upos)
        return y

    def loss(self, h, feats_gold, upos_gold):
        return _loss_and_error(self(h, upos_gold), feats_gold, self.criterion)

    def parse(self, h, upos):
        return self(h, upos).max(1)[1].cpu().numpy()

class CharLSTMEncoder(nn.Module):

    def __init__(self, field, dims, input_dropout, lstm_num_layers=2, lstm_dropout=0.33):
        super().__init__()
        dims = _field_option(field, dims)
        self.embedding = Embedding(field, (dims[0], dims[1]), input_dropout)

        if dims[2] % 2:
            raise ValueError('output_dim must be an even number.')
        self.lstm_hidden_dim = dims[2] // 2
        self.lstm = nn.LSTM(dims[1], self.lstm_hidden_dim, lstm_num_layers, dropout=lstm_dropout, bidirectional=False,
                            batch_first=True)

    def size(self):
        return self.lstm_hidden_dim * 2

    def __call__(self, instance):
        x = instance[self.field]
        x = [self.embedding(w) for w in x]
        x = rnn.pack_sequence(x, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        h = h[-1, :, :]
        return h
