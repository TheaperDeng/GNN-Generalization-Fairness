# Impact of Biased Training Node Selection

First make sure you have two folder under your pwd.

```bash
mkdir output
mkdir label_node_selection
```

Second, you need to select the biased training nodes, random training nodes, validation nodes and test nodes for each combination of {model, dataset, centrality, domain index, seed}

```bash
python selected_node_save.py --sortby degree --divide label --model GCN --dataset cora --seed 0 --domainidx 0
```

Third, run the experiment

```bash
python main.py --sortby degree --divide label --model GCN --dataset cora --seed 0 --domainidx 0
```
