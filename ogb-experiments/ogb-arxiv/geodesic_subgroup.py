import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle

device = 'cpu'
device = torch.device(device)
dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric().to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device).numpy()
test_idx = split_idx['test'].to(device).numpy()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


dis_dict = load_object("geodesic_matrix.pickle")

print(len(dis_dict))

res_dict = {}

for test in test_idx:
    min_distance = len(dis_dict[test])
    for train in train_idx:
        try:
            if dis_dict[test][train] < min_distance:
                min_distance = dis_dict[test][train]
        except:
            pass
        if min_distance != len(dis_dict[test]) and min_distance != 0:
            res_dict[test] = min_distance

sort_res = list(
    map(lambda x: x[0], sorted(res_dict.items(), key=lambda x: x[1])))
print(sorted(res_dict.items(), key=lambda x: x[1])[:100])
print(sorted(res_dict.items(), key=lambda x: x[1])[-100:])

group_num = 40
node_num_group = len(sort_res) // group_num
sub = [
    sort_res[i:i + node_num_group + 1]
    for i in range(0, len(sort_res), node_num_group + 1)
]


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


save_object(sub, "subgroup.geodesic_distance.40")
