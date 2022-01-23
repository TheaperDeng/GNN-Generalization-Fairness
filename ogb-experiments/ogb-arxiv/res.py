import os
import pickle
import argparse
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument("--noise_on_feature", action='store_true')
parser.add_argument('--type', type=str, default="agg")
args = parser.parse_args()
print(args)

# load group list
if args.type == "agg" and args.noise_on_feature:
    f_group = open("subgroup.agg.rand.40", 'rb')
else:
    f_group = open(f"subgroup.{args.type}.40", 'rb')
subgroup_list = pickle.load(f_group)

# load model state dict
if args.use_sage:
    if args.type == "agg" and args.noise_on_feature:
        f_model = open("SAGE.rand.state_dict", 'rb')
    else:
        f_model = open("SAGE.state_dict", 'rb')
else:
    if args.type == "agg" and args.noise_on_feature:
        f_model = open("GCN.rand.state_dict", 'rb')
    else:
        f_model = open("GCN.state_dict", 'rb')
state_dict = pickle.load(f_model)


# model definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


# data loading
print("loading data")
dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                 transform=T.ToSparseTensor())
print("data loaded")
device = 'cpu'
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

# add noise
if args.type == "agg" and args.noise_on_feature:
    data.x = load_object("noise.feature")

# model build
if args.use_sage:
    model = SAGE(data.num_features, args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)
else:
    model = GCN(data.num_features, args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)
model.load_state_dict(state_dict)
print(model)

# evaluate
evaluator = Evaluator(name='ogbn-arxiv')
model.eval()
out = model(data.x, data.adj_t)
res = []
for i in range(40):
    y_pred = out.argmax(dim=-1, keepdim=True)
    valid_acc = evaluator.eval({
        'y_true': data.y[subgroup_list[i]],
        'y_pred': y_pred[subgroup_list[i]],
    })['acc']
    print(valid_acc, ",")
    res.append(valid_acc)

if not os.path.exists("output"):
    os.makedirs("output")
if args.type == "agg" and args.noise_on_feature:
    subgroup_name = "agg_rand"
else:
    subgroup_name = args.type
if args.use_sage:
    model_name = "sage"
else:
    model_name = "gcn"
with open(f"output/{model_name}.{subgroup_name}", "w") as fw:
    for i in range(len(res)):
        fw.write(str(res[i]) + ",")
    fw.write("\n")
