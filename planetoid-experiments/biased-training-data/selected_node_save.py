from dgl.data import citation_graph as citegrh
from utils import importance_list_factory, subgroup_list_factory, generate_mask
import random
import networkx as nx
import numpy as np
import pickle
import torch as th

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--sortby", default="closeness", help="sort by")
parser.add_argument("--divideby", default="label", help="divide by")
parser.add_argument("--seed", default=42, help="seed")
parser.add_argument("--domainidx", default=2, help="domain idx")
parser.add_argument("--checkmethod", default=None, help="check_method")
parser.add_argument("--nodenumeach", default=20, help="node_num_each")
parser.add_argument("--dataset", default="cora", help="dataset")
parser.add_argument("--model", default="GCN", help="model")
parser.add_argument("--validnum", default=500, help="model")
parser.add_argument("--testnum", default=1000, help="test_num")
args = parser.parse_args()

# seed
random.seed(int(args.seed))
np.random.seed(int(args.seed))
th.manual_seed(int(args.seed))

# data
if args.dataset == "cora":
    data_cora = citegrh.load_cora()
if args.dataset == "pubmed":
    data_cora = citegrh.load_pubmed()
if args.dataset == "citeseer":
    data_cora = citegrh.load_citeseer()

# data prepare
graph = data_cora[0]
node_features = graph.ndata['feat'].float()  # feature
node_labels = graph.ndata['label']  # label
n_features = node_features.shape[1]  # number of feature
n_labels = int(node_labels.max().item() + 1)  # number of unique label

graph_cora = nx.Graph(graph.to_networkx().to_undirected())

# get sort list
Importance_list = importance_list_factory(graph_cora, sort_by=args.sortby)

# get group list
Group_list = subgroup_list_factory(graph_cora,
                                   node_labels,
                                   divide_by=args.divideby,
                                   dataset_name=args.dataset)

# get mask
bias_ratio = (0.1, 0.75)
train_mask, _, _ = generate_mask(Importance_list.copy(),
                                 Group_list.copy(),
                                 domain_idx=int(args.domainidx),
                                 node_num_each=int(args.nodenumeach),
                                 valid_num=int(args.validnum),
                                 test_num=int(args.testnum),
                                 bias_ratio=bias_ratio)

Importance_list_random = importance_list_factory(graph_cora, sort_by="random")

# get group list
Group_list_random = subgroup_list_factory(graph_cora,
                                          node_labels,
                                          divide_by=args.divideby,
                                          dataset_name=args.dataset)

# get mask
bias_ratio = (0.1, 0.75)
train_mask_random, _, _ = generate_mask(Importance_list_random.copy(),
                                        Group_list_random.copy(),
                                        domain_idx=int(args.domainidx),
                                        node_num_each=int(args.nodenumeach),
                                        valid_num=int(args.validnum),
                                        test_num=int(args.testnum),
                                        bias_ratio=bias_ratio)

all_idx = th.tensor(list(range(0, len(Importance_list))))

all_train = set(all_idx[train_mask].numpy()).union(
    set(all_idx[train_mask_random].numpy()))
print(all_train, len(all_train))
all_idx = list(range(0, len(Importance_list)))

# validation
remain_idx = list(filter(lambda x: x not in all_train, all_idx))
valid_idx = random.sample(remain_idx, int(args.validnum))  # 500

# test
remain_idx = list(filter(lambda x: x not in valid_idx, remain_idx))
test_idx = random.sample(remain_idx, int(args.testnum))  # 1000

# gen_bool_torch
valid_mask = th.zeros(len(Importance_list))
test_mask = th.zeros(len(Importance_list))
valid_mask[valid_idx] = 1
test_mask[test_idx] = 1
valid_mask = valid_mask.bool()
test_mask = test_mask.bool()

# save them
configs = {
    "sortby": args.sortby,
    "seed": args.seed,
    "domainidx": args.domainidx,
    "dataset": args.dataset
}

# train_mask
file = open('./label_node_selection/node_{}_train.pkl'\
            .format(str(configs)\
            .replace(" ","_")\
            .replace("{", "")\
            .replace("}","")\
            .replace("\'","")\
            .replace(":","")),
            'wb')
pickle.dump(train_mask, file)
file.close()

# train_mask_random
file = open('./label_node_selection/node_{}_train_random.pkl'\
            .format(str(configs)\
            .replace(" ","_")\
            .replace("{", "")\
            .replace("}","")\
            .replace("\'","")\
            .replace(":","")),
            'wb')
pickle.dump(train_mask_random, file)
file.close()

# valid_mask
file = open('./label_node_selection/node_{}_valid.pkl'\
            .format(str(configs)\
            .replace(" ","_")\
            .replace("{", "")\
            .replace("}","")\
            .replace("\'","")\
            .replace(":","")),
            'wb')
pickle.dump(valid_mask, file)
file.close()

# test_mask
file = open('./label_node_selection/node_{}_test.pkl'\
            .format(str(configs)\
            .replace(" ","_")\
            .replace("{", "")\
            .replace("}","")\
            .replace("\'","")\
            .replace(":","")),
            'wb')
pickle.dump(test_mask, file)
file.close()
