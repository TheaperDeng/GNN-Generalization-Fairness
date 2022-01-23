import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle
import scipy
import networkx as nx

device = torch.device("cpu")
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
test_idx = split_idx['test'].to(device)

# merge training nodes into one node
adj = data.adj_t.to_scipy().tocsr()
tr_idx = train_idx.numpy()

connected = adj[tr_idx].sum(axis=0)
connected = np.array(connected).reshape(-1)
connected[tr_idx] = 0
connected[connected > 0] = 1
connected = scipy.sparse.coo_matrix(connected)
connected_t = scipy.sparse.vstack(
    [connected.T, scipy.sparse.coo_matrix([[0]])])

aug_adj = scipy.sparse.vstack([adj.tocoo(), connected])
aug_adj = scipy.sparse.hstack([aug_adj, connected_t])

# get non-training index
ones = np.ones((adj.shape[0] + 1, ))  # +1 to include the new merged node
ones[tr_idx] = 0
te_idx = np.argwhere(ones).reshape(-1)

# removing oringinal training nodes from the adj matrix
new_adj = aug_adj.tocsr()[te_idx]
new_adj = new_adj.tocsc()[:, te_idx]

G = nx.from_scipy_sparse_matrix(new_adj)
lengths = nx.shortest_path_length(G, source=list(G.nodes)[-1])

# get a permutated index for te_idx to avoid the artifact of the original node ordering
rs = np.random.RandomState(0)
perm_idx = rs.permutation(len(te_idx[:-1]))

# convert back to the original node idx (permutated version)
res_dict = {}
fillna = len(G.nodes)
for i in perm_idx:
    if i in lengths:
        res_dict[te_idx[i]] = lengths[i]
    else:
        res_dict[te_idx[i]] = fillna

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
