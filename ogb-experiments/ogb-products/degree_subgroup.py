import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle

device = torch.device("cpu")
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
test_idx = split_idx['test'].to(device)

A = data.adj_t.to_torch_sparse_coo_tensor()
D_diag = list(torch.sparse.sum(A, dim=1))
agg_distance = {}

counter = 0
for i in test_idx:
    if counter % 1000 == 0:
        print(counter)
    agg_distance[int(i)] = int(D_diag[i])
    counter += 1

sort_res = list(
    map(lambda x: x[0], sorted(agg_distance.items(), key=lambda x: -x[1])))

group_num = 40
node_num_group = len(sort_res) // group_num

subgroup = [
    sort_res[i:i + node_num_group + 1]
    for i in range(0, len(sort_res), node_num_group + 1)
]


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


save_object(subgroup, "subgroup.degree.40")

print(list(map(len, subgroup)))