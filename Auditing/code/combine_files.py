import numpy as np
import os
from collections import defaultdict

dataset=sys.argv[1]
model=sys.argv[2]
plrv=sys.argv[3]
plrv=True if int(plrv)==1 else False
print(f"dataset: {dataset}, model: {model}, plrv: {plrv}")

# dataset = "fmnist" # "p100"
# model = "lr" # "lr"
res_dir = f"../auditing_PLRV_{plrv}/{dataset}_{model}/results"
print(res_dir)
all_nps = [f for f in os.listdir(res_dir) if f.endswith('.npy') and f.startswith('batch')]

def parse_name(fname):
    splt = fname.split('-')
    splt[9] = splt[9][:-4]
    # 0         1       2      3      4       5     6    7     8      9
    # ['batch', 'new', '300', '350', 'False', '3', '1', '1', '1.72', '1']
    return tuple([splt[v] for v in [1, 4, 5, 6, 7, 8, 9]])
print(all_nps)
print([parse_name(n) for n in all_nps])
combined = defaultdict(list)

for arr_f in all_nps:
    arr = np.load(os.path.join(res_dir, arr_f), allow_pickle=True)
    print(arr_f, parse_name(arr_f))
    combined[parse_name(arr_f)].append(arr)

for name in combined:
    print(combined[name])

for name in combined:
    print(name, np.concatenate(combined[name]).ravel().shape)
    np.save(os.path.join(res_dir, '-'.join(['bkd'] + list(name))), np.concatenate(combined[name]).ravel())
