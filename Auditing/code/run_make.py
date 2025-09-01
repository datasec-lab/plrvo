import os, sys
from itertools import product
from collections import defaultdict

dataset=sys.argv[1]
model=sys.argv[2]
plrv=sys.argv[3]
plrv=True if int(plrv)==1 else False
print(f"dataset: {dataset}, model: {model}, plrv: {plrv}")

BATCH_SIZE = 50

# dataset = "p100" # "p100"
# model = "lr" # "2f"
data_dir = f"../auditing_PLRV_{plrv}/{dataset}_{model}"
h5s = [fname for fname in os.listdir(data_dir) if fname.endswith('.h5')]

def get_cfg(h5):
    splt = h5.split('-')
    if 'no' in h5:
        return ('no', splt[1], splt[2], '.', splt[4], splt[5], splt[6])
    else:
        return ('new', splt[1], splt[2], splt[3], splt[4], splt[5], splt[6])

cfg_map = defaultdict(list)

for h5 in h5s:
    cfg_map[get_cfg(h5)].append(h5)

args = {d: len(cfg_map[d]) for d in cfg_map if len(cfg_map[d]) > 0}
# print(args)

all_exp = []

def run_exp(cmd):
    cmd = "CUDA_VISIBLE_DEVICES=0 "+cmd
    print(cmd)
    os.system(cmd)

fmt_cmd = "CUDA_VISIBLE_DEVICES=0 python make_nps.py {} {} {} {} {} {} {} {} {} {} {}"
for arg in args:
    for start in range(0, args[arg], BATCH_SIZE):
        cmd = fmt_cmd.format(arg[0], start, start + BATCH_SIZE, arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], dataset, model)
        print(cmd)
        print("wait")
        all_exp.append(cmd)
print(len(args), len(all_exp))
