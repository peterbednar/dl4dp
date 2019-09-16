
import torch.nn as nn
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

        return arc_scores, label_scores
