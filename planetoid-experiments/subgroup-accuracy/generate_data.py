import argparse
import random
import numpy as np
import pickle
import torch as th
import networkx as nx

from utils import (importance_list_factory, subgroup_list_factory,
                   generate_mask, generate_mask_number,
                   get_geodesic_distance_community,
                   get_agg_feature_distance_community)

from dgl.data import citation_graph as citegrh
from dgl.data import AmazonCoBuy
from dgl import add_self_loop

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, help="seed")
parser.add_argument("--dataset", default="cora", help="dataset")
parser.add_argument("--validnum", default=500, help="valid num")
parser.add_argument("--testnum", default=1000, help="test num")

# arguments for manipulating the training set
parser.add_argument("--sortby", default="random", help="sort by")
parser.add_argument("--divideby", default="label", help="divide by")
parser.add_argument("--domainidx", default=2, help="domain idx")
parser.add_argument("--nodenumeach", default=20, help="node num each")

# arguments for subgroups
parser.add_argument("--checkmethod", default=None, help="check method")
parser.add_argument("--noise_on_feature", default="no", help="noise on feature")

args = parser.parse_args()

# seed
random.seed(int(args.seed))
np.random.seed(int(args.seed))
th.manual_seed(int(args.seed))

# data
if args.dataset == "cora":
    data_cora = citegrh.load_cora()
elif args.dataset == "pubmed":
    data_cora = citegrh.load_pubmed()
elif args.dataset == "citeseer":
    data_cora = citegrh.load_citeseer()
elif args.dataset == "amazon_computers":
    data_cora = AmazonCoBuy('computers')
elif args.dataset == "amazon_photo":
    data_cora = AmazonCoBuy('photo')
else:
    raise NotImplementedError(f"Dataset {args.dataset} not supported.")

noise_on_feature = args.noise_on_feature

# data prepare
graph = data_cora[0]
node_features = graph.ndata['feat'].float()  # feature
if noise_on_feature == "yes":
    if args.dataset in ["cora", "pubmed", "citeseer"]:
        lamda = 5
    else:
        lamda = 1
    print(th.norm(node_features))
    tmp = th.rand(node_features.shape)
    print(th.norm(tmp))
    noise = tmp / th.norm(tmp) * th.norm(node_features)
    node_features = node_features + noise * lamda
    print(node_features)
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
if args.sortby != "number":
    train_mask, valid_mask, test_mask = generate_mask(
        Importance_list.copy(),
        Group_list.copy(),
        domain_idx=int(args.domainidx),
        node_num_each=int(args.nodenumeach),
        valid_num=int(args.validnum),
        test_num=int(args.testnum),
        bias_ratio=bias_ratio)
else:
    train_mask, valid_mask, test_mask = generate_mask_number(
        node_labels,
        Group_list.copy(),
        node_num_each_train=int(args.nodenumeach),
        domain_idx=int(args.domainidx))

# get check list
if args.checkmethod is None:
    check_list = None
if args.checkmethod in ["degree", "closeness", "betweenness", "pagerank"]:
    check_list = subgroup_list_factory(graph_cora,
                                       data_cora,
                                       divide_by=args.checkmethod,
                                       dataset_name=args.dataset)
if args.checkmethod in ["geodesic_distance"]:
    check_list = get_geodesic_distance_community(train_mask,
                                                 dataset=args.dataset,
                                                 mode=args.checkmethod,
                                                 node_num=len(
                                                     graph_cora.nodes))
if args.checkmethod in ["agg"]:
    check_list = get_agg_feature_distance_community(train_mask, graph_cora,
                                                    node_features)

# add self-loop to avoid running error
graph = add_self_loop(graph)

# save data
data_configs = {
    "sortby": args.sortby,
    "divideby": args.divideby,
    "seed": args.seed,
    "domainidx": args.domainidx,
    "checkmethod": args.checkmethod,
    "nodenumeach": args.nodenumeach,
    "dataset": args.dataset,
    "noise_on_feature": args.noise_on_feature,
    "validnum": args.validnum,
    "testnum": args.testnum
}
datafilename = "./data/{}.pkl".format(
    str(data_configs).replace(" ",
                              "_").replace("{", "").replace("}", "").replace(
                                  "\'", "").replace(":", "").replace(",", "_"))
print(datafilename)
with open(datafilename, "wb") as fw:
    pickle.dump((train_mask, valid_mask, test_mask, graph, node_features,
                 node_labels, n_features, n_labels, check_list), fw)
