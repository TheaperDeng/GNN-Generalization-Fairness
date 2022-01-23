import preprocess_utils
import argparse
import utils
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="cora", help="dataset name")
args = parser.parse_args()

# geodesic_distance
cora_matrix = preprocess_utils._get_geodesic_matrix(name=args.name)
preprocess_utils._save_object(cora_matrix,
                              f"./preprocess/{args.name}_geodesic_matrix")

# betweeness/closeness
graph = preprocess_utils._get_dgl_data(name=args.name)[0]
graph = nx.Graph(graph.to_networkx().to_undirected())
betweeness = utils.importance_list_factory(graph, sort_by='betweeness')
closeness = utils.importance_list_factory(graph, sort_by='closeness')
preprocess_utils._save_object(betweeness, f"./preprocess/{args.name}_betweeness")
preprocess_utils._save_object(closeness, f"./preprocess/{args.name}_closeness")
