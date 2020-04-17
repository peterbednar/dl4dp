import torch
import torch.nn as nn
import numpy as np

from .modules import Embeddings, LSTM, MLP, Biaffine
from .utils import tarjan

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
        self.criterion = nn.CrossEntropyLoss()
        
        self.embeddings = Embeddings(embedding_dims, input_dropout)
        input_dim = self.embeddings.size()

        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout)

        self.arc_head_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)
        self.arc_dep_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)
        self.label_head_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)
        self.label_dep_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)

        self.arc_biaffine = Biaffine(arc_mlp_dim, 1, bias_x=True, bias_y=False)
        self.label_biaffine = Biaffine(label_mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, batch):
        x = self.embeddings(batch)
        h, _ = self.lstm(x)

        arc_head = self.arc_head_mlp(h)
        arc_dep = self.arc_dep_mlp(h)
        label_head = self.label_head_mlp(h)
        label_dep = self.label_dep_mlp(h)

        arc_scores = self.arc_biaffine(arc_dep, arc_head)
        label_scores = self.label_biaffine(label_dep, label_head).permute(0, 2, 3, 1)

        return arc_scores.cpu(), label_scores.cpu()

    def loss(self, batch):
        arc_scores, label_scores = self(batch)
        indexes, _ = _get_batch_indexes(batch)

        arc_loss, arc_error = self._get_arc_loss(arc_scores, indexes)
        label_loss, label_error = self._get_label_loss(label_scores, indexes)
        return arc_loss, label_loss, arc_error, label_error

    def _get_arc_loss(self, arc_scores, indexes):
        arc_scores = arc_scores[indexes[0,:], indexes[1,:], :]
        gold_arcs = torch.from_numpy(indexes[2,:])
        return self._loss_and_error(arc_scores, gold_arcs)

    def _get_label_loss(self, label_scores, indexes):
        label_scores = label_scores[indexes[0,:], indexes[1,:], indexes[2,:], :]
        gold_labels = torch.from_numpy(indexes[3,:])
        return self._loss_and_error(label_scores, gold_labels)

    def parse(self, batch):
        arc_scores, label_scores = self(batch)
        indexes, lengths = _get_batch_indexes(batch, training=False)

        pred_arcs = self._parse_arcs(arc_scores, indexes, lengths)
        pred_labels = self._parse_labels(label_scores, indexes, pred_arcs)
        return pred_arcs, pred_labels

    def _parse_arcs(self, arc_scores, indexes, lengths):
        arc_scores = arc_scores[indexes[0,:], indexes[1,:], :].data.numpy()
        arc_pred = np.empty(arc_scores.shape[0], np.int)
        i = 0
        for k in lengths:
            scores = np.vstack([np.zeros(k+1), arc_scores[i:i+k, :k+1]]).transpose()
            heads = arc_pred[i:i+k]
            tarjan(scores, heads)
            i += k
        return arc_pred

    def _parse_labels(self, label_scores, indexes, arc_pred):
        label_scores = label_scores[indexes[0,:], indexes[1,:], arc_pred, :]
        return label_scores.data.max(1)[1].data.numpy()

    def _loss_and_error(self, scores, gold):
        count = gold.size()[0]
        pred = scores.data.max(1)[1]
        loss = self.criterion(scores, gold)
        error = (count - pred.eq(gold).sum()) / float(count)
        return loss, error

def _get_batch_indexes(batch, training=True):
    lengths = [instance.length for instance in batch]
    cols = sum(lengths)
    rows = 4 if training else 2

    i = 0
    indexes = np.empty((rows, cols), dtype=np.int64)
    for j, instance in enumerate(batch):
        for k in range(instance.length):
            indexes[0, i] = j
            indexes[1, i] = k + 1
            if training:
                indexes[2, i] = instance.head[k]
                indexes[3, i] = instance.deprel[k]
            i += 1

    return indexes, lengths
