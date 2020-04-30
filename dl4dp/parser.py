import torch
import torch.nn as nn
import numpy as np

from .modules import Embeddings, LSTM, MLP, Biaffine
from .utils import tarjan

class LSTMEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 encoder_dim, 
                 lstm_num_layers=3,
                 lstm_dropout=0.33):
        super().__init__()
        if encoder_dim % 2:
            raise ValueError('encoder_dim must be an even number.')
        lstm_hidden_dim = encoder_dim // 2
        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout)

    def forward(self, x, _):
        h, _ = self.lstm(x)
        return h

class BiaffineParser(nn.Module):

    def __init__(self,
                 embedding_dims,
                 labels_dim,
                 input_dropout=0.33,
                 encoder_dim=800,
                 arc_mlp_dim=500,
                 arc_mlp_dropout=0.33,
                 label_mlp_dim=100,
                 label_mlp_dropout=0.33,
                 **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        self.embeddings = Embeddings(embedding_dims, input_dropout)
        input_dim = self.embeddings.size()

        self.encoder = LSTMEncoder(input_dim, encoder_dim, **kwargs)

        self.arc_mlp_h = MLP(encoder_dim, arc_mlp_dim, arc_mlp_dropout)
        self.arc_mlp_d = MLP(encoder_dim, arc_mlp_dim, arc_mlp_dropout)
        self.label_mlp_h = MLP(encoder_dim, label_mlp_dim, label_mlp_dropout)
        self.label_mlp_d = MLP(encoder_dim, label_mlp_dim, label_mlp_dropout)

        self.arc_biaff = Biaffine(arc_mlp_dim, 1, bias_x=True, bias_y=False)
        self.label_biaff = Biaffine(label_mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, batch):
        lengths = [instance.length for instance in batch]

        x = self.embeddings(batch)
        h = self.encoder(x, lengths)

        arc_h = self.arc_mlp_h(h)
        arc_d = self.arc_mlp_d(h)
        label_h = self.label_mlp_h(h)
        label_d = self.label_mlp_h(h)

        arc_scores = self.arc_biaff(arc_d, arc_h).cpu()
        label_scores = self.label_biaff(label_d, label_h).permute(0, 2, 3, 1).cpu()
        return arc_scores, label_scores, lengths

    def loss(self, batch):
        arc_scores, label_scores, lengths = self(batch)
        indexes = self._get_batch_indexes(batch, lengths)

        arc_loss, arc_error = self._get_arc_loss(arc_scores, indexes)
        label_loss, label_error = self._get_label_loss(label_scores, indexes)
        loss = arc_loss + label_loss
        return loss, (arc_loss, label_loss, arc_error, label_error)

    def _get_arc_loss(self, arc_scores, indexes):
        arc_scores = arc_scores[indexes[0,:], indexes[1,:], :]
        gold_arcs = torch.from_numpy(indexes[2,:])
        return self._loss_and_error(arc_scores, gold_arcs)

    def _get_label_loss(self, label_scores, indexes):
        label_scores = label_scores[indexes[0,:], indexes[1,:], indexes[2,:], :]
        gold_labels = torch.from_numpy(indexes[3,:])
        return self._loss_and_error(label_scores, gold_labels)

    def parse(self, batch):
        if self.training:
            raise RuntimeError('Not in eval mode.')

        with torch.no_grad():
            arc_scores, label_scores, lengths = self(batch)
            indexes = self._get_batch_indexes(batch, lengths)        
            pred_arcs = self._parse_arcs(arc_scores, indexes, lengths)
            pred_labels = self._parse_labels(label_scores, indexes, pred_arcs)
            return pred_arcs, pred_labels

    def _parse_arcs(self, arc_scores, indexes, lengths):
        arc_scores = arc_scores[indexes[0,:], indexes[1,:], :].numpy()
        arc_pred = np.empty(arc_scores.shape[0], np.int)
        i = 0
        for k in lengths:
            scores = np.vstack([np.zeros(k+1), arc_scores[i:i+k, :k+1]]).transpose()
            heads = arc_pred[i:i+k]
            tarjan(scores, heads)
            i += k
        return arc_pred

    def _parse_labels(self, label_scores, indexes, pred_arcs):
        label_scores = label_scores[indexes[0,:], indexes[1,:], pred_arcs, :]
        return label_scores.max(1)[1].numpy()

    def _loss_and_error(self, scores, gold):
        pred = scores.max(1)[1]
        loss = self.criterion(scores, gold)
        error = 1 - (pred.eq(gold).sum() / float(gold.size()[0]))
        return loss, error

    def _get_batch_indexes(self, batch, lengths):
        cols = sum(lengths)
        rows = 4 if self.training else 2

        i = 0
        indexes = np.empty((rows, cols), dtype=np.int64)
        for j, instance in enumerate(batch):
            k = i + lengths[j]
            indexes[0, i:k] = j
            indexes[1, i:k] = np.arange(1, lengths[j]+1)
            if self.training:
                indexes[2, i:k] = instance.head
                indexes[3, i:k] = instance.deprel
            i = k

        return indexes
