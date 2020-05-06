import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np

class Embedding(nn.Module):

    def __init__(self, dims, input_dropout=0, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(dims[0], dims[1], padding_idx=padding_idx)
        self.input_dropout = input_dropout
        self.reset_parameters()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).long()

        if self.training and self.input_dropout > 0:
            mask = torch.rand(x.shape) > self.input_dropout
            x.masked_fill(~mask, 0)

        x = x.to(self.embedding.weight.device)
        return self.embedding(x)

    def size(self):
        return self.embedding.weight.size()[1]

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.1)
        nn.init.xavier_uniform_(self.embedding.weight, gain=gain)

def _field_option(f, opt, default=None):
    return opt.get(f, default) if isinstance(opt, dict) else opt

class Embeddings(nn.Module):

    def __init__(self, field_dims={}, input_dropout=0, padding_idx=None, opr='cat'):
        super().__init__()
        self.opr = opr
        self.embeddings = nn.ModuleDict()
        for f, dim in field_dims.items():
            self.add(f, dim, input_dropout, padding_idx, opr)

    def __getitem__(self, field):
        return self.embeddings[field]

    def __setitem__(self, field, embedding):
        self.embeddings[field] = embedding

    def size(self):
        sizes = [emb.size() for emb in self.embeddings.values()]
        return sum(sizes) if self.opr == 'cat' else max(sizes, default=0)

    def add(self, field, dim, input_dropout=0, padding_idx=None, opr='cat'):
        dim = _field_option(field, dim)
        dropout = _field_option(field, input_dropout, 0)
        padding = _field_option(field, padding_idx, None)
        self.embeddings[field] = Embeddings(dim, dropout, padding, opr) if isinstance(dim, dict) else \
                                 Embedding(dim, dropout, padding)

    def forward(self, instance):
        def _embedding(f, instance):
            embd = self.embeddings[f]
            return embd(instance) if isinstance(embd, Embeddings) else embd(instance[f])

        x = [_embedding(f, instance) for f in self.embeddings.keys()]
        if self.opr == 'cat':
            x = torch.cat(x, 1)
        elif self.opr == 'sum':
            x = torch.stack(x)
            x = torch.sum(x, 0)
        else:
            raise ValueError(f'Unknown operator {self.opr}.')
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout, bidirectional=True):
        super().__init__()
        if output_dim % 2:
            raise ValueError('output_dim must be an even number.')
        hidden_dim = output_dim // 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=bidirectional,
            batch_first=True)

    def forward(self, x):
        x = rnn.pack_sequence(x, enforce_sorted=False)
        h, c = self.lstm(x)
        h, lengths = rnn.pad_packed_sequence(h, batch_first=True)
        return h, c, lengths

class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.linear(x))
        x = self.dropout(x)
        return x

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.1)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)

class Biaffine(nn.Module):

    def __init__(self, input_size, output_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size + bias_x, input_size + bias_y))
        self.reset_parameters()
    
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.shape[:-1]).unsqueeze(-1)], -1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.shape[:-1]).unsqueeze(-1)], -1)
        x = x.unsqueeze(-3)
        y = y.unsqueeze(-3)
        s = x @ self.weight @ y.transpose(-1, -2)
        s = s.squeeze(1)
        return s

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.1)
        nn.init.xavier_uniform_(self.weight, gain=gain)
