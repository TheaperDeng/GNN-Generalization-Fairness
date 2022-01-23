# Accuracy Disparity Across Subgroups

We examine the accuracy disparity with 3 types of subgroups as described below:

1. subgroups by aggregated-feature distance

   a. With original features

   b. With noisy features

2. Subgroup by geodesic distance

3. Subgroup by node centrality

   a. Degree

   b. PageRank

   c. Betweenness

   d. Closeness

## How to run the code

For all the experiments settings:

0. Please make sure you have the following folders:

```bash
mkdir preprocess
mkdir output
mkdir data
```

1. Run `preprocess_cache.py` to cache some computation-heavy intermediate variables:

```bash
python preprocess_cache.py --name cora
```

2. Run `generate_data.py` to generate a set of meta data:

```bash
# divide by distance of agg feature with original features
python generate_data.py --seed 0 --checkmethod agg --dataset cora
# divide by distance of agg feature with randomness
python generate_data.py --seed 0 --checkmethod agg --dataset cora --noise_on_feature yes
# divide by distance on graph
python generate_data.py --seed 0 --checkmethod geodesic_distance --dataset cora
# divide by node centrality (change different centrality in --checkmethod)
python generate_data.py --seed 0 --checkmethod degree --dataset cora
```

3. Run `train_and_eval.py` to train and evaluate on the subgroups:

```bash
# divide by distance of agg feature with original features
python train_and_eval.py --seed 0 --checkmethod agg --dataset cora --model GCN
# divide by distance of agg feature with randomness
python train_and_eval.py --seed 0 --checkmethod agg --dataset cora --noise_on_feature yes  --model GCN
# divide by distance on graph
python train_and_eval.py --seed 0 --checkmethod geodesic_distance --dataset cora  --model GCN
# divide by node centrality (change different centrality in --checkmethod)
python train_and_eval.py --seed 0 --checkmethod degree --dataset cora  --model GCN
```

## Run all the experiments

Subsequently run `preprocess.sh`, `gen_data.sh`, and `run.sh` will get the results of all the experiments.
