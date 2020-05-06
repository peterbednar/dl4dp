import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from .modules import Embedding, MLP, Biaffine

class BiaffineTagger(nn.Module):

    def __init__(self):
        super().__init__()

def _loss_and_error(scores, gold, criterion):
    pred = scores.max(1)[1]
    loss = criterion(scores, gold)
    error = 1 - (pred.eq(gold).sum() / float(gold.size()[0]))
    return loss, error

class UposAffine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.affine = nn.Linear(mlp_dim, labels_dim, bias=True)

    def forward(self, h):
        x = self.mlp(h)
        y = self.affine(x)
        return y

class FeatsBiaffine(nn.Module):

    def __init__(self, input_dim, labels_dim, mlp_dim, mlp_dropout):
        super().__init__()
        self.mlp = MLP(input_dim, mlp_dim, mlp_dropout)
        self.biaffine = Biaffine(mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, h, upos):
        x = self.mlp(h)
        y = self.biaffine(x, upos)
        return y

class CharLSTMEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, lstm_num_layers=2, lstm_dropout=0.33):
        super().__init__()
        self.embedding = Embedding(input_dim)
        if output_dim % 2:
            raise ValueError('output_dim must be an even number.')
        lstm_hidden_dim = output_dim // 2
        self.lstm = nn.LSTM(input_dim[1], lstm_hidden_dim, lstm_num_layers, dropout=lstm_dropout, bidirectional=False,
                            batch_first=True)
    def __call__(self, x):
        x = [self.embedding(w) for w in x]
        x = rnn.pack_sequence(x, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        h = h[-1, :, :]
        return h
