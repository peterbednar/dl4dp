import torch
import torch.nn as nn
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

class Embeddings(nn.Module):

    def __init__(self, field_dims, input_dropout=0, padding_idx=None, opr='cat'):
        super().__init__()
        self.opr = opr
        self.embeddings = nn.ModuleDict()

        for f, dim in field_dims.items():
            dropout = input_dropout.get(f, 0) if isinstance(input_dropout, dict) else input_dropout
            padding = padding_idx.get(f, None) if isinstance(padding_idx, dict) else padding_idx
            if isinstance(dim, dict):
                self.embeddings[f] = Embeddings(dim, dropout, padding)
            else:
                self.embeddings[f] = Embedding(dim, dropout, padding)

    def forward(self, instances):
        batch = []
        for instance in instances:
            x = [self.embeddings[f](instance[f]) for f in self.embeddings.keys()]
            if self.opr == 'cat':
                x = torch.cat(x, 1)
            if self.opr == 'sum':
                x = torch.stack(x)
                x = torch.sum(x, 0)
            batch.append(x)
        return batch

    def size(self):
        sizes = [emb.size() for emb in self.embeddings.values()]
        return sum(sizes) if self.opr == 'cat' else max(sizes, 0)

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
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        s = x @ self.weight @ y.transpose(-1, -2)
        s = s.squeeze(1)
        return s

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.1)
        nn.init.xavier_uniform_(self.weight, gain=gain)
