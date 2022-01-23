import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle
import networkx as nx

device = 'cpu'
device = torch.device(device)
dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric().to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device).numpy()
test_idx = split_idx['test'].to(device).numpy()

A = data.adj_t.to_torch_sparse_coo_tensor()
list_A_coo = A.coalesce().indices()
print(type(list_A_coo))
numpy_A_coo = list_A_coo.numpy().transpose()
print(numpy_A_coo.shape)

G = nx.Graph()
for i in range(numpy_A_coo.shape[0]):
    G.add_edge(numpy_A_coo[i, 0], numpy_A_coo[i, 1])
print(type(G))

distance_dict = {}
for test in test_idx:
    distance_matrix = nx.shortest_path_length(G, source=test)
    distance_dict[test] = dict(distance_matrix)


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


save_object(distance_dict, "geodesic_matrix.pickle")
