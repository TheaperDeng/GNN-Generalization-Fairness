import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--noise_on_feature", action='store_true')
args = parser.parse_args()
print(args)


def load_object(filename):
    f = open(filename, 'rb')
    object_to_load = pickle.load(f)
    f.close()
    return object_to_load


def save_object(object_to_save, filename):
    f = open(filename, 'wb')
    pickle.dump(object_to_save, f)
    f.close()


distance_agg = {}
group_num = 40

for start in range(0, 2213091, 2213091 // 50):
    end = start + 2213091 // 50 if start + 2213091 // 50 < 2213091 else 2213091
    if args.noise_on_feature:
        filename = f"cache/subgroup.agg_{start}_{end}.rand.40"
    else:
        filename = f"cache/subgroup.agg_{start}_{end}.40"
    sub_dict = load_object(filename)
    distance_agg = {**distance_agg, **sub_dict}

sort_res = list(
    map(lambda x: x[0], sorted(distance_agg.items(), key=lambda x: x[1])))
node_num_group = len(sort_res) // group_num
res = [
    sort_res[i:i + node_num_group + 1]
    for i in range(0, len(sort_res), node_num_group + 1)
]

print(list(map(len, res)))

if args.noise_on_feature:
    save_object(res, f"subgroup.agg.40")
else:
    save_object(res, f"subgroup.agg.rand.40")
