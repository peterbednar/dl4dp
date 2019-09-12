
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np

class Embedding(nn.Module):

    def __init__(self, dims, input_dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(dims[0], dims[1])
        self.input_dropout = input_dropout
        self.reset_parameters()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).long()

        if self.training and self.input_dropout > 0:
            mask = torch.rand(x.shape) > self.input_dropout
            x = x * mask.long()

        return self.embedding(x)

    def size(self):
        return self.embedding.weight.shape

    def reset_parameters(self):
        gain = nn.init.calculate_gain("leaky_relu", 0.1)
        nn.init.xavier_uniform_(self.embedding.weight, gain=gain)

class Embeddings(nn.Module):

    def __init__(self, field_dims, input_dropout=0):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for f, dim in field_dims.items():
            dropout = input_dropout.get(f, 0) if isinstance(input_dropout, dict) else input_dropout
            self.embeddings[f] = Embedding(dim, dropout)

    def forward(self, instances):
        batch = []
        for instance in instances:
            x = [self.embeddings[f](instance[f]) for f in self.embeddings.keys()]
            x = torch.cat(x, 1)
            batch.append(x)
        return batch

    def size(self):
        return sum(embed.size()[1] for embed in self.embeddings.values())

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0):
        super().__init__()
        self.root = nn.Parameter(torch.Tensor(1, input_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True, batch_first=True)

    def forward(self, batch):
        for i, x in enumerate(batch):
            batch[i] = torch.cat([self.root, x])
        x = rnn.pack_sequence(batch)
        h, _ = self.lstm(x)
        h, _ = rnn.pad_packed_sequence(h, batch_first=True)
        return h

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
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

class Bilinear(nn.Module):
    
    def __init__(self, input_size, output_size, bias_x=True, bias_y=True, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size, input_size))
        self.register_parameter("bias_x", nn.Parameter(torch.Tensor(output_size, input_size)) if bias_x else None)
        self.register_parameter("bias_y", nn.Parameter(torch.Tensor(output_size, input_size)) if bias_x else None)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(output_size)) if bias_x else None)
        self.reset_parameters()
    
    def forward(self, x, y):
        s = F.bilinear(x, y, self.weight, self.bias)
        if self.bias_x is not None:
            s += F.linear(x, self.bias_x)
        if self.bias_y is not None:
            s += F.linear(y, self.bias_y)
        return s

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        for param in self.parameters():
            nn.init.uniform_(param, -bound, bound)

class Biaffine(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self, x, y):
        return x