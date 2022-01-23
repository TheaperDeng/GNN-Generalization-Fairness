import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle
from torch_geometric.nn import GCNConv

import argparse


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, help="Start position.")
parser.add_argument("--end", type=int, help="End position.")
parser.add_argument("--noise_on_feature", action='store_true')
args = parser.parse_args()
print(args)

start = args.start
end = args.end

device = torch.device("cpu")
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
test_idx = split_idx['test'].to(device)

if args.noise_on_feature:
    data.x = load_object("noise.feature")

print("test_idx len:", test_idx.shape[0])

feat_dim = data.x.size(1)

conv = GCNConv(feat_dim, feat_dim, cached=True, bias=False)
conv.lin.weight = torch.nn.Parameter(torch.eye(feat_dim))
conv.to(device)

with torch.no_grad():
    agg = conv(data.x, data.adj_t)
    agg = conv(agg, data.adj_t)

print(type(agg), agg.shape)

agg = agg.numpy()
group_num = 40

agg_distance = {}
train_idx = list(train_idx)
test_idx = list(test_idx)
train_agg = agg[train_idx]

for i in test_idx[start:end]:
    distance_tmp = train_agg - agg[i]
    agg_distance[int(i)] = float(min(np.linalg.norm(distance_tmp, axis=1)))

if args.noise_on_feature:
    save_object(
        agg_distance,
        f"cache/subgroup.agg_{start}_{end}.rand.40"
    )
else:
    save_object(
        agg_distance,
        f"cache/subgroup.agg_{start}_{end}.40"
    )
