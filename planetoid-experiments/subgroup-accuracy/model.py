import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv, SGConv, APPNPConv


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


# https://github.com/dmlc/dgl/blob/master/examples/pytorch/appnp/appnp.py
class APPNP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hiddens=[64],
                 activation=F.relu,
                 feat_drop=0.5,
                 edge_drop=0.5,
                 alpha=0.1,
                 k=10,
                 **config):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], out_feats))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(graph, h)
        return h


class SGC(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_units=32,
                 dropout=0.5,
                 activation="relu"):
        super(SGC, self).__init__()
        self.sgc = SGConv(in_feats, out_feats, k=2)

    def forward(self, graph, inputs):
        x = self.sgc(graph, inputs)
        return F.log_softmax(x, dim=1)
