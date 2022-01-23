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

tmp = torch.rand(data.x.shape)
noise = tmp / torch.norm(tmp) * torch.norm(data.x)
data.x = data.x + noise * 20
save_object(data.x, "noise.feature")
