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

        self.tags = nn.ModuleDict()
        self.tags['upos'] = UposAffine(encoder_dim, output_dims['upos'], upos_mlp_dim, upos_mlp_dropout) 
        for f, dim in output_dims['feats']:
            self.tags[f] = FeatsBiaffine(encoder_dim, dim, feats_mlp_dim, feats_mlp_dropout)

    def forward(self, batch):
        x = [self.embeddings(instance) for instance in batch]
        h = self.encoder(x, unpad=True)
        return h

    def loss(self, batch):
        h = self(batch)
        upos_gold = _get_upos(batch)
        return self.tags['upos'].loss(h, upos_gold)

    def _get_upos(self, batch):
        x = [torch.from_numpy(t['upos']) for t in batch]
        return torch.cat(x, -1)

class UposAffine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.affine = nn.Linear(mlp_dim, labels_dim, bias=False)

    def forward(self, h):
        x = self.mlp(h)
        y = self.affine(x)
        return y

    def loss(self, h, upos_gold):
        return loss_and_error(self(h), upos_gold)

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

    def __init__(self, field, dims, input_dropout, lstm_num_layers=2, lstm_dropout=0.33):
        super().__init__()
        dims = _field_option(field, dims)
        self.embedding = Embedding(field, (dims[0], dims[1][0]), input_dropout)
        self.lstm_hidden_dim = dims[1][1]
        self.lstm = nn.LSTM(dims[1], self.lstm_hidden_dim, lstm_num_layers, dropout=lstm_dropout, bidirectional=True,
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
