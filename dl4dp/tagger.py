import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from .modules import loss_and_error
from .modules import Embedding, Embeddings, MLP, LSTM, Bilinear, _field_option

class BiaffineTagger(nn.Module):

    def __init__(self,
                 input_dims,
                 output_dims,
                 input_dropout=0.33,
                 char_lstm_num_layers=2,
                 char_lstm_dropout=0.33,
                 lstm_hidden_dim=200,
                 lstm_num_layers=2,
                 lstm_dropout=0.33,
                 upos_mlp_dim=100,
                 upos_mlp_dropout=0.33,
                 feats_mlp_dim=50,
                 feats_mlp_dropout=0.33):
        super().__init__()

        self.embeddings = Embeddings('cat')
        self.embeddings['form'] = Embedding('form', input_dims, input_dropout)
        self.embeddings['form:chars'] = CharLSTMEncoder('form:chars', input_dims, input_dropout,
                char_lstm_num_layers, char_lstm_dropout)

        input_dim = self.embeddings.size()
        self.encoder = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, True)

        encoder_dim = lstm_hidden_dim * 2
        upos_dim = output_dims['upos']

        self.tags = nn.ModuleDict()
        self.tags['upos'] = Affine(encoder_dim, upos_dim, upos_mlp_dim, upos_mlp_dropout) 
        for f, dim in output_dims['feats'].items():
            self.tags[f] = FeatsBiaffine(encoder_dim, dim, feats_mlp_dim, feats_mlp_dropout)
        self.upos_embedding = Embedding('upos', (upos_dim, feats_mlp_dim))

    def forward(self, batch):
        x = [self.embeddings(instance) for instance in batch]
        h, _, _ = self.encoder(x, unpad=True)
        h = torch.cat(h)
        return h

    def loss(self, batch):
        h = self(batch)
        upos_gold = self._get_gold('upos', batch)
        loss, error = self.tags['upos'].loss(h, upos_gold)
        return loss, (error, )

    def _get_gold(self, field, batch):
        return torch.cat([torch.from_numpy(instance[field]) for instance in batch])

class Affine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.affine = nn.Linear(mlp_dim, labels_dim, bias=False)

    def forward(self, h):
        x = self.mlp(h)
        y = self.affine(x)
        return y

    def loss(self, h, gold):
        return loss_and_error(self(h), gold)

    def parse(self, h):
        return self(h).max(1)[1]

class FeatsBiaffine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.bilinear = Bilinear(mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, h, upos):
        x = self.mlp(h)
        y = self.bilinear(x, upos)
        return y

    def loss(self, h, feats_gold, upos_gold):
        return loss_and_error(self(h, upos_gold), feats_gold)

    def parse(self, h, upos):
        return self(h, upos).max(1)[1]

class CharLSTMEncoder(nn.Module):

    def __init__(self, field, dims, input_dropout, num_layers=2, lstm_dropout=0.33):
        super().__init__()
        self.field = field
        dims = _field_option(field, dims)

        input_dim = dims[1][0]
        self.hidden_dim = dims[1][1]
        self.bidirectional = True

        self.embedding = Embedding(field, (dims[0], input_dim), input_dropout)
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers, dropout=lstm_dropout,
                            bidirectional=self.bidirectional, batch_first=True)

    def size(self):
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

    def __call__(self, instance):
        x = instance[self.field]
        x = [self.embedding(w) for w in x]
        x = rnn.pack_sequence(x, enforce_sorted=False)
        _, (h, _) = self.lstm(x)

        if self.bidirectional:
            h = torch.cat((h[-1, :, :], h[-2, :, :]), -1)
        else:
            h = h[-1, :, :]
        return h
