from dgl.data import citation_graph as citegrh
from utils import load_object, save_result, load_result, train_model
import random
import networkx as nx
from model import GCN, MLP, GAT
import numpy as np
import torch

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--sortby", default="closeness", help="sort by")
parser.add_argument("--divideby", default="label", help="divide by")
parser.add_argument("--seed", default=42, help="data seed")
parser.add_argument("--domainidx", default=2, help="domain idx")
parser.add_argument("--checkmethod", default=None, help="check_method")
parser.add_argument("--nodenumeach", default=20, help="node_num_each")
parser.add_argument("--dataset", default="cora", help="dataset")
parser.add_argument("--model", default="GCN", help="model")
parser.add_argument("--validnum", default=500, help="model")
parser.add_argument("--testnum", default=1000, help="test_num")
args = parser.parse_args()

# prepare all configs
all_configs = {
    "sortby": args.sortby,
    "divideby": args.divideby,
    "seed": args.seed,
    "domainidx": args.domainidx,
    "checkmethod": args.checkmethod,
    "nodenumeach": args.nodenumeach,
    "dataset": args.dataset,
    "model": args.model,
    "validnum": args.validnum,
    "testnum": args.testnum
}
print(all_configs)

# seed
random.seed(int(args.seed))
np.random.seed(int(args.seed))
torch.manual_seed(int(args.seed))

# model
if args.model == "GCN":
    modelclass = GCN
if args.model == "GAT":
    modelclass = GAT
if args.model == "MLP":
    modelclass = MLP

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

if args.sortby in ["degree", "closeness", "betweenness", "pagerank"]:
    configs = {
        "sortby": args.sortby,
        "seed": args.seed,
        "domainidx": args.domainidx,
        "dataset": args.dataset
    }
    train_mask = load_object('./label_node_selection/node_{}_train.pkl'\
                             .format(str(configs)\
                             .replace(" ","_")\
                             .replace("{", "")\
                             .replace("}","")\
                             .replace("\'","")\
                             .replace(":","")))
    valid_mask = load_object('./label_node_selection/node_{}_valid.pkl'\
                             .format(str(configs)\
                             .replace(" ","_")\
                             .replace("{", "")\
                             .replace("}","")\
                             .replace("\'","")\
                             .replace(":","")))
    test_mask = load_object('./label_node_selection/node_{}_test.pkl'\
                             .format(str(configs)\
                             .replace(" ","_")\
                             .replace("{", "")\
                             .replace("}","")\
                             .replace("\'","")\
                             .replace(":","")))
rand_train_mask = load_object('./label_node_selection/node_{}_train_random.pkl'\
                             .format(str(configs)\
                             .replace(" ","_")\
                             .replace("{", "")\
                             .replace("}","")\
                             .replace("\'","")\
                             .replace(":","")))

# get check list
check_list = None

# train prepare
best_test_acc, tpr, fpr = train_model(modelclass,
                                      train_mask,
                                      valid_mask,
                                      test_mask,
                                      graph,
                                      node_features,
                                      node_labels,
                                      n_features,
                                      n_labels,
                                      epoch_num=400,
                                      groups=check_list)

filename = "./output/res_{}.json".format(
    str(all_configs).replace(" ", "_").replace("{",
                                               "").replace("}", "").replace(
                                                   "\'", "").replace(":", ""))

best_test_acc_rand, tpr_rand, fpr_rand = train_model(modelclass,
                                                     rand_train_mask,
                                                     valid_mask,
                                                     test_mask,
                                                     graph,
                                                     node_features,
                                                     node_labels,
                                                     n_features,
                                                     n_labels,
                                                     epoch_num=400,
                                                     groups=check_list)

filename_rand = "./output/res_{}_rand.json".format(
    str(all_configs).replace(" ", "_").replace("{",
                                               "").replace("}", "").replace(
                                                   "\'", "").replace(":", ""))

# save
save_result(best_test_acc, tpr, fpr, filename)
save_result(best_test_acc_rand, tpr_rand, fpr_rand, filename_rand)

# load
res = load_result(filename)
res_rand = load_result(filename_rand)

# print
print("Manipulated Training Set FPR: ", res["fpr"])
print("Not Manipulated Training Set FPR: ", res_rand["fpr"])
print("Relative Ratio of FPR:", [res["fpr"][i]/res_rand["fpr"][i] for i in range(len(res["fpr"]))])
