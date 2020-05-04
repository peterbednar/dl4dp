import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from .modules import Embedding

class CharLSTMEncoder(nn.Module):

    def __init__(self, input_dims, output_dim, lstm_num_layers=3, lstm_dropout=0.33):
        super().__init__()
        self.embedding = Embedding(input_dims)
        if output_dim % 2:
            raise ValueError('output_dim must be an even number.')
        lstm_hidden_dim = output_dim // 2
        self.lstm = nn.LSTM(input_dims[1], lstm_hidden_dim, lstm_num_layers, dropout=lstm_dropout, bidirectional=False,
                            batch_first=True)
    def __call__(self, x):
        x = [self.embedding(w) for w in x]
        x = rnn.pack_sequence(x, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        h = h[-1, :, :]
        return h
