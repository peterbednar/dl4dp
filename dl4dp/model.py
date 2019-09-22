
import math

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from .modules import Embeddings, LSTM, MLP, Biaffine

class BiaffineParser(nn.Module):
    
    def __init__(self, embedding_dims, labels_dim, input_dropout=0.33,
                lstm_hidden_dim=400, lstm_num_layers=3, lstm_dropout=0.33,
                arc_mlp_dim=500, arc_mlp_dropout=0.33,
                label_mlp_dim=100, label_mlp_dropout=0.33):
        super().__init__()
        
        self.embeddings = Embeddings(embedding_dims, input_dropout)
        input_dim = self.embeddings.size()

        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout)

        self.arc_head_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)
        self.arc_dep_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)

        self.label_head_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)
        self.label_dep_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)

        self.arc_biaffine = Biaffine(arc_mlp_dim, 1)
        self.label_biaffine = Biaffine(label_mlp_dim, labels_dim)

    def forward(self, instances):
        x = self.embeddings(instances)
        h, batch_lengths = self.lstm(x)

        arc_head = self.arc_head_mlp(h)
        arc_dep = self.arc_dep_mlp(h)
        arc_scores = self.arc_biaffine(arc_dep, arc_head)

        label_head = self.label_head_mlp(h)
        label_dep = self.label_dep_mlp(h)
        label_scores = self.label_biaffine(label_dep, label_head).permute(0, 2, 3, 1)

        return arc_scores, label_scores, batch_lengths

    def _get_arc_loss(self, arc_scores, gold_arcs, mask):
        arc_scores.masked_fill(~mask.unsqueeze(1), -math.inf)
        arc_scores.masked_fill(torch.eye(arc_scores.size(1)).unsqueeze(0) > 0, -math.inf)

    def _get_label_loss(self, label_scores, gold_arcs, gold_labels, mask):
        pass

    def _get_mask(self, max_length, batch_lengths):
        return torch.arange(max_length)[None, :] < batch_lengths[:, None]

    def _get_gold_arcs(self, instances):
        heads = [torch.from_numpy(instance["head"]).long() for instance in instances]
        return rnn.pad_sequence(heads, batch_first=True)

    def _get_gold_labels(self, instances):
        deps = [torch.from_numpy(instance["deprel"]).long() for instance in instances]
        return rnn.pad_sequence(deps, batch_first=True)
