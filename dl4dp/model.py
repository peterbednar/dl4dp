import torch
import torch.nn as nn
import numpy as np
from .modules import Embeddings, LSTM, MLP, Biaffine

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
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        
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
        batch.sort(key=lambda x: x.length, reverse=True)

        x = self.embeddings(batch)
        h, _ = self.lstm(x)

        arc_head = self.arc_head_mlp(h)
        arc_dep = self.arc_dep_mlp(h)
        arc_scores = self.arc_biaffine(arc_dep, arc_head)

        label_head = self.label_head_mlp(h)
        label_dep = self.label_dep_mlp(h)
        label_scores = self.label_biaffine(label_dep, label_head).permute(0, 2, 3, 1)

        return arc_scores, label_scores

    def loss(self, batch):
        arc_scores, label_scores = self(batch)

        indexes = _get_batch_indexes(batch)
        gold_arcs = _get_gold_arcs(batch)
        gold_labels = _get_gold_labels(batch)

        arc_loss = self._get_arc_loss(arc_scores, gold_arcs, indexes)
        label_loss = self._get_label_loss(label_scores, gold_arcs, gold_labels, indexes)

        return arc_loss, label_loss

    def _get_arc_loss(self, arc_scores, gold_arcs, indexes):
        arc_scores = arc_scores[indexes[:,0], indexes[:,1], :]
        return self.criterion(arc_scores, gold_arcs)

    def _get_label_loss(self, label_scores, gold_arcs, gold_labels, indexes):
        label_scores = label_scores[indexes[:,0], indexes[:,1], indexes[:,2], :]
        return self.criterion(label_scores, gold_labels)

def _get_batch_indexes(batch):
    num_words = sum([instance.length for instance in batch])
    indexes = np.empty((num_words, 3), dtype=np.int32)
    i = 0
    for j, instance in enumerate(batch):
        for dep in range(instance.length):
            indexes[i, 0] = j
            indexes[i, 1] = dep + 1
            indexes[i, 2] = instance.head[dep]
            i += 1
    return indexes

def _get_gold_arcs(instances):
    heads = [torch.from_numpy(instance.head).long() for instance in instances]
    return torch.cat(heads, -1)

def _get_gold_labels(instances):
    deps = [torch.from_numpy(instance.deprel).long() for instance in instances]
    return torch.cat(deps, -1)
