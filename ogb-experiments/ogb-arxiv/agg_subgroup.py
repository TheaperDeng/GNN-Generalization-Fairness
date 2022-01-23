import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle
import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
parser.add_argument("--noise_on_feature", action='store_true')
args = parser.parse_args()


def get_agg_feature_distance_community(adj_matrix,
                                       feature,
                                       train_idx,
                                       test_idx,
                                       group_num=5):
    print(type(adj_matrix))
    A = adj_matrix  # torch_sparse.tensor.SparseTensor
    print("complete A")
    A = A + torch.sparse_coo_tensor(
        [[i for i in range(169343)], [i for i in range(169343)]], [1] * 169343)
    print("complete A+I")
    D_diag = list(torch.sparse.sum(A, dim=1))
    print("complete D_diag")
    D_1 = [1 / x for x in D_diag]
    D_1 = torch.sparse_coo_tensor(
        [[i for i in range(169343)], [i for i in range(169343)]], D_1)
    print("complete D_1")

    agg = torch.sparse.mm(D_1, A)
    print("complete mm1")
    agg = torch.sparse.mm(agg, D_1)
    print("complete mm2")
    agg = torch.sparse.mm(agg, A).to_dense().numpy()
    print("complete mm3")
    agg = np.matmul(agg, feature)
    print("complete mm4")

    agg_distance = {}
    train_idx = list(train_idx)
    for k in range(len(test_idx)):
        i = test_idx[k]
        if k % 100 == 0:
            print(k)
        agg_distance[i] = float('inf')
        for j in train_idx:
            agg_distance[i] = min(agg_distance[i],
                                  np.linalg.norm(agg[i] - agg[j]))

    sort_res = list(
        map(lambda x: x[0], sorted(agg_distance.items(), key=lambda x: x[1])))
    node_num_group = len(sort_res) // group_num
    return [
        sort_res[i:i + node_num_group + 1]
        for i in range(0, len(sort_res), node_num_group + 1)
    ]


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


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

if args.noise_on_feature:
    data.x = load_object(f"noise.feature")

agg_group = get_agg_feature_distance_community(
    adj_matrix=data.adj_t.to_torch_sparse_coo_tensor(),
    feature=data.x.numpy(),
    train_idx=train_idx,
    test_idx=test_idx,
    group_num=40)
if args.noise_on_feature:
    save_object(agg_group, "subgroup.agg.rand.40")
else:
    save_object(agg_group, "subgroup.agg.40")
