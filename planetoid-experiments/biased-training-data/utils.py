import numpy as np
from dgl.data import citation_graph as citegrh
import torch as th
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import json

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

import pickle


def importance_list_factory(graph, sort_by="degree"):
    '''
    :param graph: networkx graph
    :param sort_by: One of "random", "number", "betweenness", "closeness", "degree", "pagerank"
    :return: a list of node index sorted by sort_by in descending order.
    Note: "random" and "number" all return the same random list
    '''
    if sort_by == "degree":
        res = list(
            map(
                lambda x: x[0],
                sorted(nx.degree_centrality(graph).items(),
                       key=lambda x: -x[1])))
    if sort_by == "closeness":
        res = list(
            map(
                lambda x: x[0],
                sorted(nx.closeness_centrality(graph).items(),
                       key=lambda x: -x[1])))
    if sort_by == "betweenness":
        res = list(
            map(
                lambda x: x[0],
                sorted(nx.betweenness_centrality(graph).items(),
                       key=lambda x: -x[1])))
    if sort_by == "pagerank":
        res = list(
            map(lambda x: x[0],
                sorted(nx.pagerank(graph).items(), key=lambda x: -x[1])))
    if sort_by == "random" or sort_by == "number":
        res = list(range(len(graph.nodes)))
        random.shuffle(res)
    return res


def subgroup_list_factory(graph,
                          labels,
                          divide_by="label",
                          dataset_name="cora"):
    '''
    :param graph: networkx graph
    :param data_cora: dgl.data
    :param divide_by: One of "label", 
           "degree", "closeness", "betweenness", "pagerank", "community_raw[deprecated]"
    :param dataset_name: dataset name
    '''
    if divide_by == "label":
        # label list of list
        label_list = [[] for j in range(len(np.unique(labels)))]
        for node in range(len(graph.nodes)):
            label_list[labels[node]].append(node)
        return label_list


def generate_mask(node_importance,
                  node_subgroup,
                  node_num_each=20,
                  domain_idx=2,
                  valid_num=500,
                  test_num=1000,
                  bias_ratio=(0.1, 0.75)):
    '''
    Generate Mask for biased or not biased.
    :param node_importance: got from importance_list_factory
    :param node_subgroup: got from subgroup_list_factory
    :param node_num_each:
    :param domain_idx:
    :param valid_num:
    :param test_num:
    :param bias_ratio:
    '''
    all_idx = list(range(0, len(node_importance)))
    train_idx = []
    valid_idx = []
    test_idx = []

    # training
    node_importance = dict(zip(node_importance, range(0,
                                                      len(node_importance))))
    domain_group = node_subgroup.pop(domain_idx)
    train_idx += random.sample(
        sorted(domain_group,
               key=lambda x: node_importance[x])[:int(bias_ratio[0] *
                                                      len(domain_group))],
        min(int(node_num_each * bias_ratio[1]),
            int(bias_ratio[0] * len(domain_group))))
    train_idx += random.sample(
        sorted(domain_group,
               key=lambda x: node_importance[x])[int(bias_ratio[0] *
                                                     len(domain_group)):],
        node_num_each - min(int(node_num_each * bias_ratio[1]),
                            int(bias_ratio[0] * len(domain_group))))
    for subgroup in node_subgroup:
        train_idx += random.sample(
            sorted(subgroup,
                   key=lambda x: -node_importance[x])[:int(bias_ratio[0] *
                                                           len(subgroup))],
            min(int(node_num_each * bias_ratio[1]),
                int(bias_ratio[0] * len(subgroup))))
        train_idx += random.sample(
            sorted(subgroup,
                   key=lambda x: -node_importance[x])[int(bias_ratio[0] *
                                                          len(subgroup)):],
            node_num_each - min(int(node_num_each * bias_ratio[1]),
                                int(bias_ratio[0] * len(subgroup))))

    # validation
    remain_idx = list(filter(lambda x: x not in train_idx, all_idx))
    valid_idx = random.sample(remain_idx, valid_num)  # 500

    # test
    remain_idx = list(filter(lambda x: x not in valid_idx, remain_idx))
    test_idx = random.sample(remain_idx, test_num)  # 1000

    # gen_bool_torch
    train_mask = th.zeros(len(node_importance))
    valid_mask = th.zeros(len(node_importance))
    test_mask = th.zeros(len(node_importance))
    train_mask[train_idx] = 1
    valid_mask[valid_idx] = 1
    test_mask[test_idx] = 1
    train_mask = train_mask.bool()
    valid_mask = valid_mask.bool()
    test_mask = test_mask.bool()

    # print
    print("split result:")
    print(th.sum(train_mask), th.sum(valid_mask), th.sum(test_mask))
    return train_mask, valid_mask, test_mask


def correction_helper(model, graph, node_features, mask, node_labels):
    model.eval()
    with th.no_grad():
        logits = model(graph, node_features)
        logits = logits[mask]
        _, indices = th.max(logits, dim=1)
        node_labels_tmp = node_labels[mask]
        correct = th.sum(indices == node_labels_tmp)
    acc = correct.item() * 1.0 / len(node_labels_tmp)
    return acc, node_labels_tmp, indices


def evaluate(model, node_labels, mask, graph, node_features, groups=None):
    acc, node_labels_new, indices = correction_helper(model, graph,
                                                      node_features, mask,
                                                      node_labels)
    if groups is None:
        # evaluate on label group
        tpr_list = []
        fpr_list = []
        for label in range(int(th.max(node_labels_new)) + 1):
            if float(th.sum(node_labels_new == label)) != 0:
                tpr_list.append(
                    float(
                        th.sum((indices == label)
                               & (node_labels_new == label))) /
                    float(th.sum(node_labels_new == label)))
            else:
                tpr_list.append(1)
            if float(th.sum(indices == label)) != 0:
                fpr_list.append(
                    float(
                        th.sum((indices == label)
                               & (node_labels_new != label))) /
                    float(th.sum(node_labels_new != label)))
            else:
                fpr_list.append(0)
        return acc, tpr_list, fpr_list
    else:
        # evaluate on customized group
        tpr = []
        model.eval()
        with th.no_grad():
            logits = model(graph, node_features)
            _, indices = th.max(logits, dim=1)
        for group in groups:
            mask_group = th.zeros(len(node_labels))
            mask_group[group] = 1
            mask_group = mask_group.bool()
            new_mask = (mask & mask_group)
            if sum(new_mask) == 0:
                tpr.append(1)
                continue
            indices_g = indices[new_mask]
            node_labels_tmp = node_labels[new_mask]
            correct = th.sum(indices_g == node_labels_tmp)
            acc_group = correct.item() * 1.0 / len(node_labels_tmp)
            tpr.append(acc_group)
        return acc, tpr, None


def train_model(model_class,
                train_mask,
                valid_mask,
                test_mask,
                graph,
                node_features,
                node_labels,
                n_features,
                n_labels,
                epoch_num=1000,
                groups=None):
    '''
    :param model_class: One of MLP, GAT, GCN
    :param train_mask: get from generate_mask
    :param valid_mask: get from generate_mask
    :param test_mask: get from generate_mask
    :param graph: get from dgl.data
    :param node_features:
    :param node_labels:
    :param n_features:
    :param n_labels:
    :param epoch_num:
    :param groups: checkmethod
    '''
    # print(type(node_features))
    best_valid_acc = 0
    best_test_acc = 0
    tpr = []
    fpr = []
    model = model_class(in_feats=n_features,
                        out_feats=n_labels,
                        n_units=16,
                        dropout=0.5)
    opt = th.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epoch_num):
        model.train()
        logits = model(graph, node_features)
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc, _, _ = evaluate(model,
                             node_labels,
                             valid_mask,
                             graph,
                             node_features,
                             groups=groups)
        if (epoch + 1) % (epoch_num // 10) == 1:
            print("epoch:{}/{} loss:".format(epoch, epoch_num), loss.item())
            print("valid acc:", acc)
        if best_valid_acc < acc:
            best_valid_acc = acc
            acc_test, tpr_tmp, fpr_tmp = evaluate(model,
                                                  node_labels,
                                                  test_mask,
                                                  graph,
                                                  node_features,
                                                  groups=groups)
            tpr = tpr_tmp
            fpr = fpr_tmp
            best_test_acc = acc_test
    print(best_test_acc)
    print(tpr)
    print(fpr)
    return best_test_acc, tpr, fpr


def save_result(acc, tpr, fpr, filename):
    res = {"acc": acc, "tpr": tpr, "fpr": fpr}
    res_str = json.dumps(res)
    f = open(filename, 'w')
    f.write(res_str)
    f.close()


def load_result(filename):
    f = open(filename, 'r')
    content = f.read()
    a = json.loads(content)
    f.close()
    return a


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load