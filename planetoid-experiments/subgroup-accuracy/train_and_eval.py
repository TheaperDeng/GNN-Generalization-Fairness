import argparse
from model import GCN, MLP, GAT, SGC, APPNP
from utils import train_model, load_result, save_result
import pickle

parser = argparse.ArgumentParser()
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

# arguments for model training
parser.add_argument("--seed", default=0, help="seed")
parser.add_argument("--model", default="GCN", help="model")

args = parser.parse_args()

# prepare all configs
configs = {
    "sortby": args.sortby,
    "divideby": args.divideby,
    "seed": args.seed,
    "domainidx": args.domainidx,
    "checkmethod": args.checkmethod,
    "nodenumeach": args.nodenumeach,
    "dataset": args.dataset,
    "model": args.model,
    "noise_on_feature": args.noise_on_feature,
    "validnum": args.validnum,
    "testnum": args.testnum
}
print(configs)

# model
if args.model == "GCN":
    modelclass = GCN
elif args.model == "GAT":
    modelclass = GAT
elif args.model == "MLP":
    modelclass = MLP
elif args.model == "APPNP":
    modelclass = APPNP
elif args.model == "SGC":
    modelclass = SGC
else:
    raise NotImplementedError(f"Model {args.model} not supported.")

# load data
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

with open(datafilename, "rb") as fr:
    train_mask, valid_mask, test_mask, graph, node_features, node_labels, n_features, n_labels, check_list = pickle.load(
        fr)

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
                                      groups=check_list,
                                      seed=args.seed)

filename = "./output/res___{}.json".format(
    str(configs).replace(" ", "_").replace("{", "").replace("}", "").replace(
        "\'", "").replace(":", "").replace(",", "_"))
save_result(best_test_acc, tpr, fpr, filename)

res = load_result(filename)

# print
print("subgroup acc (from nearest to the furthest):", res["tpr"])
