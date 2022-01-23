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

print("train:", len(train_idx))
print("test:", len(test_idx))

A = data.adj_t.to_torch_sparse_coo_tensor()
A = A + torch.sparse_coo_tensor(
    [[i for i in range(169343)], [i for i in range(169343)]], [1] * 169343)
D_diag = list(torch.sparse.sum(A, dim=1))
agg_distance = {}
for i in test_idx:
    agg_distance[i] = D_diag[i]
sort_res = list(
    map(lambda x: x[0], sorted(agg_distance.items(), key=lambda x: -x[1])))

group_num = 40
node_num_group = len(sort_res) // group_num

subgroup = [
    sort_res[i:i + node_num_group + 1]
    for i in range(0, len(sort_res), node_num_group + 1)
]
score_list = list(
    map(lambda x: x[1], sorted(agg_distance.items(), key=lambda x: -x[1])))
print(score_list[0:10])


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


save_object(subgroup, "subgroup.degree.40")

print(list(map(len, subgroup)))
