import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import pickle


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")
dataset = PygNodePropPredDataset(name='ogbn-products',
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
test_idx = split_idx['test'].to(device)

tmp = torch.rand(data.x.shape)
noise = tmp / torch.norm(tmp) * torch.norm(data.x)
data.x = data.x + noise

save_object(data.x, "noise.feature")
