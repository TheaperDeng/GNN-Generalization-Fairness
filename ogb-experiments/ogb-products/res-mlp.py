import argparse
import os
import pickle

import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def train(model, x, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    out = model(x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (MLP)')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--noise_on_feature", action='store_true')
    args = parser.parse_args()
    print(args)

    if args.noise_on_feature:
        f_group = open(f"subgroup.agg.rand.40", 'rb')
        f_model = open(f"MLP.rand.state_dict", 'rb')
    else:
        f_group = open(f"subgroup.agg.40", 'rb')
        f_model = open(f"MLP.state_dict", 'rb')

    device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = data.x
    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)
    x = x.to(device)

    if args.noise_on_feature:
        x = load_object("noise.feature")

    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes,
                args.num_layers, args.dropout).to(device)

    subgroup_list = pickle.load(f_group)
    state_dict = pickle.load(f_model)

    model.load_state_dict(state_dict)
    print(model)

    evaluator = Evaluator(name='ogbn-products')
    model.eval()
    out = model(x)
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
    if args.noise_on_feature:
        subgroup_name = "agg_rand"
    else:
        subgroup_name = "agg"
    model_name = "mlp"
    with open(f"output/{model_name}.{subgroup_name}", "w") as fw:
        for i in range(len(res)):
            fw.write(str(res[i]) + ",")
        fw.write("\n")


if __name__ == "__main__":
    main()
