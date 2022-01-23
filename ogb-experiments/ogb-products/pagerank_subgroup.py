import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle
import networkx as nx

device = torch.device('cpu')
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric().to(device)
A = data.adj_t.to_torch_sparse_coo_tensor()
list_A_coo = A.coalesce().indices()
print(type(list_A_coo))
numpy_A_coo = list_A_coo.numpy().transpose()
print(numpy_A_coo.shape)

G = nx.Graph()
for i in range(numpy_A_coo.shape[0]):
    G.add_edge(numpy_A_coo[i, 0], numpy_A_coo[i, 1])
print(type(G))


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


split_idx = dataset.get_idx_split()
all_test = split_idx['test'].to(device)

all_sorted_list = sorted(nx.pagerank(G).items(), key=lambda x: -x[1])
sorted_list = []
for x in all_sorted_list:
    if x[0] in all_test:
        sorted_list.append(int(x[0]))

group_num = 40
num_each = len(sorted_list) // group_num
grouped_idx = [
    sorted_list[i:i + num_each + 1]
    for i in range(0, len(sorted_list), num_each + 1)
]

print(list(map(len, grouped_idx)))


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


save_object(grouped_idx, "subgroup.pagerank.40")
