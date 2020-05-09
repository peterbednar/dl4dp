import torch
import torch.nn as nn
import numpy as np
from itertools import accumulate

from .modules import loss_and_error, unbind_sequence
from .modules import Embedding, Embeddings, MLP, Biaffine, LSTM
from .utils import tarjan

class BiaffineParser(nn.Module):

    def __init__(self,
                 input_dims,
                 output_dims,
                 input_dropout=0.33,
                 lstm_hidden_dim=400,
                 arc_mlp_dim=400,
                 arc_mlp_dropout=0.33,
                 label_mlp_dim=100,
                 label_mlp_dropout=0.33,
                 **kwargs):
        super().__init__()

        self.embeddings = Embeddings('cat')
        self.embeddings['form'] = Embedding('form', input_dims, input_dropout)
        self.embeddings['upos_feats'] = Embeddings('sum', 'upos_feats', input_dims, input_dropout, 1)

        input_dim = self.embeddings.size()
        label_dim = output_dims['deprel']

        self.encoder = WordLSTMEncoder(input_dim, lstm_hidden_dim, **kwargs)

        self.arc_biaff = ArcBiaffine(lstm_hidden_dim*2, arc_mlp_dim, arc_mlp_dropout)
        self.lab_biaff = LabelBiaffine(lstm_hidden_dim*2, label_dim, label_mlp_dim, label_mlp_dropout)

    def forward(self, batch):
        indexes, lengths = self._get_batch_indexes(batch)
        x = [self.embeddings(instance) for instance in batch]
        h = self.encoder(x)
        return h, indexes, lengths

    def loss(self, batch):
        h, indexes, _ = self(batch)
        arc_loss, arc_error = self.arc_biaff.loss(h, indexes)
        lab_loss, lab_error = self.lab_biaff.loss(h, indexes)
        loss = arc_loss + lab_loss
        return loss, (arc_loss, lab_loss, arc_error, lab_error)

    def parse(self, batch, unbind=True):
        if self.training:
            raise RuntimeError('Not in eval mode.')

        with torch.no_grad():
            h, indexes, lengths = self(batch)
            pred_arcs = self.arc_biaff.parse(h, indexes, lengths)
            pred_labs = self.lab_biaff.parse(h, indexes, pred_arcs)
            if unbind:
                pred_arcs = unbind_sequence(pred_arcs, lengths)
                pred_labs = unbind_sequence(pred_labs, lengths)
            return {'head': pred_arcs, 'deprel': pred_labs}

    def _get_batch_indexes(self, batch):
        lengths = [x.length for x in batch]
        cols = sum(lengths)
        rows = 4 if self.training else 2
        indexes = torch.empty((rows, cols), dtype=torch.long)

        for i, k in enumerate(accumulate(lengths)):
            j = k - lengths[i]
            indexes[0,j:k] = i
            indexes[1,j:k] = torch.arange(1, lengths[i]+1)
            if self.training:
                indexes[2,j:k] = torch.from_numpy(batch[i].head)
                indexes[3,j:k] = torch.from_numpy(batch[i].deprel)

        return indexes, lengths

class ArcBiaffine(nn.Module):

    def __init__(self,
                encoder_dim,
                mlp_dim,
                mlp_dropout):
        super().__init__()
        self.h_mlp = MLP(encoder_dim, mlp_dim, mlp_dropout)
        self.d_mlp = MLP(encoder_dim, mlp_dim, mlp_dropout)
        self.biaffine = Biaffine(mlp_dim, 1, bias_x=True, bias_y=False)

    def forward(self, h):
        arc_h = self.h_mlp(h)
        arc_d = self.d_mlp(h)
        arc_scores = self.biaffine(arc_h, arc_d).cpu()
        return arc_scores

    def loss(self, h, indexes):
        arc_scores = self(h)
        arc_scores = arc_scores[indexes[0,:], indexes[1,:], :]
        gold_arcs = indexes[2,:]
        return loss_and_error(arc_scores, gold_arcs)

    def parse(self, h, indexes, lengths):
        arc_scores = self(h)
        arc_scores = arc_scores[indexes[0,:], indexes[1,:], :].numpy()
        arc_pred = np.empty(arc_scores.shape[0], np.int64)
        i = 0
        for k in lengths:
            scores = arc_scores[i:i+k, :k+1]
            heads = arc_pred[i:i+k]
            tarjan(scores, heads)
            i += k
        arc_pred = torch.from_numpy(arc_pred)
        return arc_pred

class LabelBiaffine(nn.Module):

    def __init__(self,
                encoder_dim,
                labels_dim,
                mlp_dim,
                mlp_dropout):
        super().__init__()
        self.h_mlp = MLP(encoder_dim, mlp_dim, mlp_dropout)
        self.d_mlp = MLP(encoder_dim, mlp_dim, mlp_dropout)
        self.biaffine = Biaffine(mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, h):
        lab_h = self.h_mlp(h)
        lab_d = self.d_mlp(h)
        lab_scores = self.biaffine(lab_h, lab_d).permute(0, 2, 3, 1).cpu()
        return lab_scores

    def loss(self, h, indexes):
        lab_scores = self(h)
        lab_scores = lab_scores[indexes[0,:], indexes[1,:], indexes[2,:], :]
        lab_gold = indexes[3,:]
        return loss_and_error(lab_scores, lab_gold)

    def parse(self, h, indexes, pred_arcs):
        lab_scores = self(h)
        lab_scores = lab_scores[indexes[0,:], indexes[1,:], pred_arcs, :]
        return lab_scores.max(1)[1]

class WordLSTMEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 lstm_hidden_dim, 
                 lstm_num_layers=3,
                 lstm_dropout=0.33):
        super().__init__()
        self.root = nn.Parameter(torch.empty(input_dim))
        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, dropout=lstm_dropout, bidirectional=True)
        self.reset_parameters()

    def forward(self, batch):
        batch = [torch.cat((self.root.unsqueeze(0), x)) for x in batch]
        h, _, _ = self.lstm(batch)
        return h

    def reset_parameters(self):
        nn.init.uniform_(self.root)
