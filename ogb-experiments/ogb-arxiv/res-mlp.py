import os
import pickle
import argparse
import torch
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

parser = argparse.ArgumentParser(description='OGBN-Arxiv (MLP)')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument("--noise_on_feature", action='store_true')
args = parser.parse_args()
print(args)

if args.noise_on_feature:
    f_group = open("subgroup.agg.rand.40", 'rb')
else:
    f_group = open(f"subgroup.agg.40", 'rb')

if args.noise_on_feature:
    f_model = open("MLP.rand.state_dict", 'rb')
else:
    f_model = open("MLP.state_dict", 'rb')


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


subgroup_list = pickle.load(f_group)
state_dict = pickle.load(f_model)

device = 'cpu'
dataset = PygNodePropPredDataset(name='ogbn-arxiv')
split_idx = dataset.get_idx_split()
data = dataset[0]

x = data.x

if args.noise_on_feature:
    x = load_object("noise.feature")
x = x.to(device)

y_true = data.y.to(device)
train_idx = split_idx['train'].to(device)

model = MLP(x.size(-1), args.hidden_channels, dataset.num_classes,
            args.num_layers, args.dropout).to(device)
model.load_state_dict(state_dict)
print(model)

evaluator = Evaluator(name='ogbn-arxiv')
model.eval()
out = model(x)
res = []
for i in range(40):
    y_pred = out.argmax(dim=-1, keepdim=True)
    valid_acc = evaluator.eval({
        'y_true': y_true[subgroup_list[i]],
        'y_pred': y_pred[subgroup_list[i]],
    })['acc']
    print(str(valid_acc) + ",")
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
