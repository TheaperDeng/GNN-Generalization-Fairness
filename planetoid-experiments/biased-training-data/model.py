import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_units=32,
                 dropout=0.5,
                 activation="relu"):
        super(GCN, self).__init__()
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.gc1 = GraphConv(in_feats, n_units, activation=self.activation)
        self.gc2 = GraphConv(n_units, out_feats)
        self.dropout = dropout

    def forward(self, graph, inputs):
        x = self.gc1(graph, inputs)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(graph, x)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_units=8,
                 num_heads=8,
                 activation="elu",
                 dropout=0.6,
                 negative_slope=0.2):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList()
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.gat_layers.append(
            GATConv(in_feats, n_units, num_heads, dropout, dropout,
                    negative_slope, False, self.activation))
        self.gat_layers.append(
            GATConv(n_units * num_heads, out_feats, 1, dropout, dropout,
                    negative_slope, False, None))

    def forward(self, graph, inputs):
        h = self.gat_layers[0](graph, inputs).flatten(1)
        logits = self.gat_layers[1](graph, h).mean(1)
        return F.log_softmax(logits, dim=1)


class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 n_units,
                 out_feats,
                 dropout=None,
                 activation="relu"):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, n_units)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_units, out_feats)

    def forward(self, graph, x):
        return self.linear2(self.relu(self.linear1(x)))
