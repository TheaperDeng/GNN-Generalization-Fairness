# OGB-Products Dataset Experiment

In this folder, you may conduct experience on OGB-Products dataset for 4 experience settings:

1. subgroups by aggregated-feature distance

   a. With original features

   b. With noisy features

2. Subgroup by geodesic distance

3. Subgroup by node centrality

## Quick Start

0. Cache the noisy version of data

```bash
python generate_noisy_data.py
```

1. Cache the subgroups

```bash
# cache for aggregated-feature subgroup
sh agg_parts.sh  # can be parallelized
python agg_subgroup_gather.py  # aggregated-feature distance on original features
python agg_subgroup_gather.py --noise_on_feature  # aggregated-feature distance on noisy features

# cache for node centrality subgroup
python degree_subgroup.py
python pagerank_subgroup.py

# cache for geodesic distance subgroup
python deodesic_subgroup.py
```

2. Train the models

```bash
# Training on original features
python gnn.py  # GCN
python gnn.py --use_sage  # SAGE
python mlp.py  # MLP

# Training on noisy features
python gnn.py --noise_on_feature  # GCN
python gnn.py --use_sage --noise_on_feature  # SAGE
python mlp.py --noise_on_feature  # MLP
```

3. Print the result

```bash
# GCN and SAGE
python res.py --type agg # GCN, subgroup by aggregated-feature distance on original features
python res.py --type agg --noise_on_feature  # MLP, subgroup by aggregated-feature distance on noisy features
python res.py --use_sage --type agg  # SAGE, subgroup by aggregated-feature distance on original features
python res.py --use_sage --type agg --noise_on_feature  # SAGE, subgroup by aggregated-feature distance on noisy features
python res.py --type geodesic_distance # GCN, subgroup by geodesic distance
python res.py --use_sage --type geodesic_distance # SAGE, subgroup by geodesic distance
python res.py --type degree # GCN, subgroup by degree centrality
python res.py --use_sage --type degree # SAGE, subgroup by degree centrality
python res.py --type pagerank # GCN, subgroup by pagerank centrality
python res.py --use_sage --type pagerank # SAGE, subgroup by pagerank centrality

# MLP
python res-mlp.py  # MLP, subgroup by aggregated-feature distance on original features
python res-mlp.py --noise_on_feature  # MLP, subgroup by aggregated-feature distance on noisy features
```

