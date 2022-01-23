import numpy as np
import torch as th
import random
import torch.nn.functional as F
import json

import networkx as nx

import pickle


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


def importance_list_factory(graph, sort_by="degree", dataset_name="cora"):
    '''
    :param graph: networkx graph
    :param sort_by: One of "random", "number", "betweeness", "closeness", "degree", "pagerank"
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
        try:
            res = load_object('./preprocess/{dataset_name}_closeness')
            print("loading closeness")
        except:
            print("generating closeness")
            res = list(
                map(
                    lambda x: x[0],
                    sorted(nx.closeness_centrality(graph).items(),
                        key=lambda x: -x[1])))
    if sort_by == "betweeness":
        try:
            res = load_object('./preprocess/{dataset_name}_betweeness')
            print("loading betweeness")
        except:
            print("generating betweeness")
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


def get_agg_feature_distance_community(train_mask,
                                       nx_graph,
                                       feature,
                                       group_num=5):
    A = nx.adj_matrix(nx_graph).A
    A = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A, axis=1))
    D_1 = np.linalg.inv(D)

    agg = np.matmul(np.matmul(np.matmul(np.matmul(D_1, A), D_1), A), feature)

    agg_distance = {}
    train_idx = th.tensor(list(range(0, A.shape[0])))
    train_idx = list(train_idx[train_mask].numpy())
    for i in range(A.shape[0]):
        agg_distance[i] = float('inf')
        for j in train_idx:
            agg_distance[i] = min(agg_distance[i],
                                  np.linalg.norm(agg[i] - agg[j]))

    sort_res = list(
        map(lambda x: x[0], sorted(agg_distance.items(), key=lambda x: x[1])))
    # score_res = list(
    #     map(lambda x: x[1], sorted(agg_distance.items(), key=lambda x: x[1])))
    # save_object(score_res, "agg.score")
    # save_object(sort_res, "agg.idx")
    node_num_group = len(sort_res) // group_num
    return [
        sort_res[i:i + node_num_group + 1]
        for i in range(0, len(sort_res), node_num_group + 1)
    ]


def subgroup_list_factory(graph,
                          labels,
                          divide_by="label",
                          dataset_name="cora"):
    '''
    :param graph: networkx graph
    :param data_cora: dgl.data
    :param divide_by: One of "label", "degree", "closeness", "betweeness", "pagerank"
    :param dataset_name: dataset name
    '''
    if divide_by == "label":
        # label list of list
        label_list = [[] for j in range(len(np.unique(labels)))]
        for node in range(len(graph.nodes)):
            label_list[labels[node]].append(node)
        return label_list
    if divide_by in ["degree", "closeness", "betweeness", "pagerank"]:
        # return 10 subgroup with decreasing order under these strategy
        sorted_list = importance_list_factory(graph, sort_by=divide_by, dataset_name=dataset_name)
        group_num = 10
        num_each = len(sorted_list) // group_num
        grouped_idx = [
            sorted_list[i:i + num_each + 1]
            for i in range(0, len(sorted_list), num_each + 1)
        ]
        return grouped_idx


def get_geodesic_distance_community(train_mask,
                                    dataset='cora',
                                    group_num=5,
                                    mode='geodesic_distance',
                                    node_num=2708):
    all_idx = th.tensor(list(range(0, node_num)))
    train_idx = list(all_idx[train_mask].numpy())
    matrix = load_object(f"./preprocess/{dataset}_geodesic_matrix")
    if mode == 'geodesic_distance':
        min_distance_list = {}
        for i in range(node_num):
            min_distance = node_num
            for j in train_idx:
                try:
                    min_distance = min(min_distance, matrix[i][j])
                except:
                    pass
            if min_distance != node_num and min_distance != 0:
                min_distance_list[i] = min_distance
        sort_res = list(
            map(lambda x: x[0],
                sorted(min_distance_list.items(), key=lambda x: x[1])))
    node_num_group = len(sort_res) // group_num
    return [
        sort_res[i:i + node_num_group + 1]
        for i in range(0, len(sort_res), node_num_group + 1)
    ]


def generate_mask_number(node_labels,
                         node_subgroup,
                         node_num_each_train=20,
                         domain_idx=2,
                         node_num_each_valid=20,
                         node_num_each_test=50):
    '''
    Generate Mask for biased or not biased.
    '''
    all_idx = th.tensor(list(range(0, len(node_labels))))
    train_idx = []
    valid_idx = []
    test_idx = []
    domain_group = node_subgroup.pop(domain_idx)

    # training
    for label in np.unique(node_labels):
        dom_label_group = set(
            all_idx[node_labels == label].numpy()).intersection(
                set(domain_group))
        train_idx += random.sample(
            dom_label_group, min(node_num_each_train, len(dom_label_group)))
        print("label " + str(label),
              len(set(train_idx).intersection(set(domain_group))))
    rest = set(domain_group) - set(train_idx)
    train_idx += random.sample(
        rest,
        node_num_each_train * len(np.unique(node_labels)) -
        len(set(train_idx).intersection(set(domain_group))))
    print(len(train_idx))

    # validation
    remain_idx = list(filter(lambda x: x not in train_idx, all_idx))
    group_remain = set(list(map(int,
                                remain_idx))).intersection(set(domain_group))
    valid_idx += random.sample(group_remain, node_num_each_valid)
    for group in np.unique(node_subgroup):
        group_remain = set(list(map(int, remain_idx))).intersection(set(group))
        valid_idx += random.sample(group_remain, node_num_each_valid)

    # test
    remain_idx = list(filter(lambda x: x not in valid_idx, remain_idx))
    group_remain = set(list(map(int,
                                remain_idx))).intersection(set(domain_group))
    test_idx += random.sample(group_remain, node_num_each_test)
    for group in np.unique(node_subgroup):
        group_remain = set(list(map(int, remain_idx))).intersection(set(group))
        test_idx += random.sample(group_remain, node_num_each_test)

    # check
    test_com_groups = []
    for group in [domain_group] + node_subgroup:
        test_com_groups.append(
            set(group).intersection(set(list(map(int, test_idx)))))
    print(list(map(len, test_com_groups)))

    # gen_bool_torch
    train_mask = th.zeros(len(node_labels))
    valid_mask = th.zeros(len(node_labels))
    test_mask = th.zeros(len(node_labels))
    train_mask[train_idx] = 1
    valid_mask[valid_idx] = 1
    test_mask[test_idx] = 1
    train_mask = train_mask.bool()
    valid_mask = valid_mask.bool()
    test_mask = test_mask.bool()
    print(th.sum(train_mask), th.sum(valid_mask), th.sum(test_mask))
    return train_mask, valid_mask, test_mask


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
                    float(th.sum(indices == label)))
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
                groups=None,
                seed=42):
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
    random.seed(int(seed))
    np.random.seed(int(seed))
    th.manual_seed(int(seed))

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
