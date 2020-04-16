import math

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from .modules import Embeddings, LSTM, MLP, Biaffine

def _get_mask(batch_lengths):
    max_length = batch_lengths.max().item()
    return torch.arange(max_length)[None, :] < batch_lengths[:, None]

def _get_gold_arcs(instances):
    heads = [torch.from_numpy(instance.head).long() for instance in instances]
    return rnn.pad_sequence(heads, batch_first=True)

def _get_gold_labels(instances):
    deps = [torch.from_numpy(instance.deprel).long() for instance in instances]
    return rnn.pad_sequence(deps, batch_first=True)

class BiaffineParser(nn.Module):
    
    def __init__(self,
                 embedding_dims,
                 labels_dim,
                 input_dropout=0.33,
                 lstm_hidden_dim=400,
                 lstm_num_layers=3,
                 lstm_dropout=0.33,
                 arc_mlp_dim=500,
                 arc_mlp_dropout=0.33,
                 label_mlp_dim=100,
                 label_mlp_dropout=0.33):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.embeddings = Embeddings(embedding_dims, input_dropout)
        input_dim = self.embeddings.size()

        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout)

        self.arc_head_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)
        self.arc_dep_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)

        self.label_head_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)
        self.label_dep_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)

        self.arc_biaffine = Biaffine(arc_mlp_dim, 1)
        self.label_biaffine = Biaffine(label_mlp_dim, labels_dim)

    def forward(self, batch):
        batch.sort(key=len, reverse=True)

        x = self.embeddings(batch)
        h, batch_lengths = self.lstm(x)

        arc_head = self.arc_head_mlp(h)
        arc_dep = self.arc_dep_mlp(h)
        arc_scores = self.arc_biaffine(arc_dep, arc_head)

        label_head = self.label_head_mlp(h)
        label_dep = self.label_dep_mlp(h)
        label_scores = self.label_biaffine(label_dep, label_head).permute(0, 2, 3, 1)

        return arc_scores, label_scores, batch_lengths

    def loss(self, batch):
        arc_scores, label_scores, batch_lengths = self(batch)

        mask = _get_mask(batch_lengths)
        gold_arcs = _get_gold_arcs(batch)
        gold_labels = _get_gold_labels(batch)

        arc_loss = self._get_arc_loss(arc_scores, gold_arcs, mask)
        label_loss = self._get_label_loss(label_scores, gold_arcs, gold_labels, mask)

        return arc_loss, label_loss

    def _get_arc_loss(self, arc_scores, gold_arcs, mask):
        pass

    def _get_label_loss(self, label_scores, gold_arcs, gold_labels, mask):
        pass
