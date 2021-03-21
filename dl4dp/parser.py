import torch
import torch.nn as nn
from itertools import accumulate

from .modules import loss_and_error, unbind_sequence
from .modules import Embedding, Embeddings, MLP, Bilinear, Biaffine, LSTM
from .utils import tarjan


class BiaffineParser(nn.Module):

    def __init__(self,
                 input_dims,
                 output_dims,
                 input_dropout=0.33,
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
        self.encoder = WordLSTMEncoder(input_dim, **kwargs)

        encoder_dim = self.encoder.size()
        label_dim = output_dims['deprel']

        self.arc_biaff = ArcBiaffine(encoder_dim, arc_mlp_dim, arc_mlp_dropout)
        self.lab_biaff = LabelBiaffine(encoder_dim, label_dim, label_mlp_dim, label_mlp_dropout)

    def forward(self, batch):
        x = [self.embeddings(instance) for instance in batch]
        h = self.encoder(x)
        indexes, lengths = self._get_batch_indexes(batch, h.device)
        return h, indexes, lengths

    def loss(self, batch):
        h, indexes, _ = self(batch)
        arc_loss, arc_error = self.arc_biaff.loss(h, indexes)
        lab_loss, lab_error = self.lab_biaff.loss(h, indexes)
        loss = arc_loss + lab_loss
        metrics = {
            'head_loss': arc_loss,
            'deprel_loss': lab_loss,
            'head_error': arc_error,
            'deprel_error': lab_error
        }
        return loss, metrics

    def parse(self, batch, unbind=True, device=None):
        if self.training:
            raise RuntimeError('not in eval mode')

        with torch.no_grad():
            h, indexes, lengths = self(batch)
            pred_arcs = self.arc_biaff.parse(h, indexes, lengths)
            pred_labs = self.lab_biaff.parse(h, indexes, pred_arcs)

            if device is not None:
                device = torch.device(device)
                pred_arcs = pred_arcs.to(device)
                pred_labs = pred_labs.to(device)

            if unbind:
                pred_arcs = unbind_sequence(pred_arcs, lengths)
                pred_labs = unbind_sequence(pred_labs, lengths)

            return {'head': pred_arcs, 'deprel': pred_labs}

    def _get_batch_indexes(self, batch, device):
        lengths = [x.length for x in batch]
        cols = sum(lengths)
        rows = 4 if self.training else 2

        indexes = torch.empty((rows, cols), dtype=torch.long)
        if torch.cuda.is_available():
            indexes = indexes.pin_memory()

        for i, k in enumerate(accumulate(lengths)):
            j = k - lengths[i]
            indexes[0, j:k] = i
            indexes[1, j:k] = torch.arange(1, lengths[i]+1)
            if self.training:
                indexes[2, j:k] = torch.from_numpy(batch[i].head)
                indexes[3, j:k] = torch.from_numpy(batch[i].deprel)

        indexes = indexes.to(device, non_blocking=True)
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

    def forward(self, h, indexes):
        arc_h = self.h_mlp(h)
        arc_d = self.d_mlp(h)
        arc_scores = self.biaffine(arc_d, arc_h)
        arc_scores = arc_scores[indexes[0, :], indexes[1, :], :]
        return arc_scores

    def loss(self, h, indexes):
        arc_scores = self(h, indexes)
        gold_arcs = indexes[2, :]
        return loss_and_error(arc_scores, gold_arcs)

    def parse(self, h, indexes, lengths):
        arc_scores = self(h, indexes).cpu()
        arc_pred = torch.empty(arc_scores.shape[0], dtype=torch.long)
        if torch.cuda.is_available():
            arc_pred = arc_pred.pin_memory()
        i = 0
        for k in lengths:
            scores = arc_scores[i:i+k, :k+1]
            heads = arc_pred[i:i+k]
            tarjan(scores, heads)
            i += k
        arc_pred = arc_pred.to(indexes.device, non_blocking=True)
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
        self.bilinear = Bilinear(mlp_dim, labels_dim, bias_x=True, bias_y=True)

    def forward(self, h, indexes, heads):
        lab_h = h[indexes[0, :], heads, :]
        lab_d = h[indexes[0, :], indexes[1, :], :]
        lab_h = self.h_mlp(lab_h)
        lab_d = self.d_mlp(lab_d)
        lab_scores = self.bilinear(lab_d, lab_h)
        return lab_scores

    def loss(self, h, indexes):
        lab_scores = self(h, indexes, indexes[2, :])
        lab_gold = indexes[3, :]
        return loss_and_error(lab_scores, lab_gold)

    def parse(self, h, indexes, pred_arcs):
        lab_scores = self(h, indexes, pred_arcs)
        return lab_scores.max(1)[1]


class WordLSTMEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 lstm_hidden_dim=400,
                 lstm_num_layers=3,
                 lstm_dropout=0.33):
        super().__init__()
        self.root = nn.Parameter(torch.empty(input_dim))
        self.hidden_dim = lstm_hidden_dim
        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, dropout=lstm_dropout, bidirectional=True)
        self.reset_parameters()

    def size(self):
        return self.hidden_dim * 2

    def forward(self, batch):
        batch = [torch.cat((self.root.unsqueeze(0), x)) for x in batch]
        h, _, _ = self.lstm(batch)
        return h

    def reset_parameters(self):
        nn.init.uniform_(self.root)
