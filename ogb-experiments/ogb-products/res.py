import argparse
import os

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import pickle
import time


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


def train(model, data, train_idx, optimizer):
    start_time = time.time()
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    print(time.time() - start_time)

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument("--noise_on_feature", action='store_true')
    parser.add_argument("--type",
                        default="agg",
                        type=str,
                        help="experiment type")
    args = parser.parse_args()
    print(args)

    if args.type == "agg":
        if args.noise_on_feature:
            f_group = open(f"subgroup.agg.rand.40", 'rb')
        else:
            f_group = open(f"subgroup.agg.40", 'rb')
    if args.type == "degree" or args.type == "pagerank":
        f_group = open(f"subgroup.{args.type}.40", 'rb')

    device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        if args.noise_on_feature:
            f_model = open(f"SAGE.rand.state_dict", 'rb')
        else:
            f_model = open(f"SAGE.state_dict", 'rb')
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        if args.noise_on_feature:
            f_model = open(f"GCN.rand.state_dict", 'rb')
        else:
            f_model = open(f"GCN.state_dict", 'rb')
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    if args.noise_on_feature:
        data.x = load_object("noise.feature")

    subgroup_list = pickle.load(f_group)
    state_dict = pickle.load(f_model)

    model.load_state_dict(state_dict)
    print(model)

    evaluator = Evaluator(name='ogbn-products')
    model.eval()
    out = model(data.x, data.adj_t)
    res = []
    for i in range(len(subgroup_list)):
        y_pred = out.argmax(dim=-1, keepdim=True)
        valid_acc = evaluator.eval({
            'y_true': data.y[subgroup_list[i]],
            'y_pred': y_pred[subgroup_list[i]],
        })['acc']
        # print(len(subgroup_list[i]))
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


if __name__ == "__main__":
    main()
