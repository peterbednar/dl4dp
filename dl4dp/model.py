
import torch.nn as nn
from .modules import Embeddings, LSTM, MLP, Biaffine

class BiaffineParser(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()

        embedding_dims = kwargs.get("embedding_dims")
        input_dropout = kwargs.get("input_dropout", 0)
        self.embeddings = Embeddings(embedding_dims, input_dropout)
        input_dim = self.embeddings.size()

        lstm_hidden_dim = kwargs.get("lstm_hidden_dim", 100)
        lstm_num_layers = kwargs.get("lstm_num_layers", 2)
        lstm_dropout = kwargs.get("lstm_dropout", 0)

        self.lstm = LSTM(input_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout)

        arc_mlp_dim = kwargs.get("arc_mlp_dim", 100)
        arc_mlp_dropout = kwargs.get("arc_mlp_dropout", 0)
        self.arc_head_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)
        self.arc_dep_mlp = MLP(lstm_hidden_dim * 2, arc_mlp_dim, arc_mlp_dropout)

        label_mlp_dim = kwargs.get("label_mlp_dim", 100)
        label_mlp_dropout = kwargs.get("label_mlp_dropout", 0)
        self.label_head_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)
        self.label_dep_mlp = MLP(lstm_hidden_dim * 2, label_mlp_dim, label_mlp_dropout)

        self.arc_biaffine = Biaffine(arc_mlp_dim, 1)

        labels_dim = kwargs.get("labels_dim")
        self.label_biaffine = Biaffine(label_mlp_dim, labels_dim)

    def forward(self, instances):
        x = self.embeddings(instances)
        h = self.lstm(x)

        arc_head = self.arc_head_mlp(h)
        arc_dep = self.arc_dep_mlp(h)
        arc_scores = self.arc_biaffine(arc_head, arc_dep)

        label_head = self.label_head_mlp(h)
        label_dep = self.label_dep_mlp(h)
        label_scores = self.label_biaffine(label_head, label_dep)

        return arc_scores, label_scores
