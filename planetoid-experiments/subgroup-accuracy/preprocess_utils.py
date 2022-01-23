import networkx as nx
import pickle
import json


def _get_dgl_data(name="cora"):
    from dgl.data import citation_graph as citegrh
    from dgl.data import AmazonCoBuy
    if name == "cora":
        return citegrh.load_cora()
    if name == "pubmed":
        return citegrh.load_pubmed()
    if name == "citeseer":
        return citegrh.load_citeseer()
    if name == "amazon_computers":
        return AmazonCoBuy('computers')
    if name == "amazon_photo":
        return AmazonCoBuy('photo')
    return None


def _get_geodesic_matrix(name="cora"):
    _dgl_data = _get_dgl_data(name)
    _dgl_graph = _dgl_data[0]
    _graph = nx.Graph(_dgl_graph.to_networkx().to_undirected())
    geodesic_matrix = nx.shortest_path_length(_graph)
    geodesic_matrix = dict(geodesic_matrix)
    return geodesic_matrix


def _save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def _load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


def _save_dict(dict_to_save, filename):
    dict_str = json.dumps(dict_to_save)
    f = open(filename, 'w')
    f.write(dict_str)
    f.close()


def _load_dict(filename):
    f = open(filename, 'r')
    content = f.read()
    dict_to_load = json.loads(content)
    f.close()
    return dict_to_load