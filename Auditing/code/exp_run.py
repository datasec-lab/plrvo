from itertools import product
import os, sys

dataset=sys.argv[1]
model=sys.argv[2]
plrv=sys.argv[3]
plrv=True if int(plrv)==1 else False
print(f"dataset: {dataset}, model: {model}, plrv: {plrv}")

pois_ct = 1
clip_norms = [0.1, 0.3, 0.4, 1, 1, 1, 0]
noises_p100 = [14.02, 6.04, 3.27, 1.86, 1.58, 1.40, 0]
noises_fmnist = [18.09, 7.78, 4.19, 2.33, 1.96, 1.72, 0]
epses= [0.2, 0.5, 1, 2, 2.5, 3, "inf"]
init_mult = 1
bkd_start, bkd_trials = 0, 500
# dataset = "fmnist" # "p100"
old_bkd = False
# model = "2f" # "lr"
data_dir = "datasets"
save_dir = f"../auditing_PLRV_{plrv}/{dataset}_{model}"
plrv_paths = [
    f"code/plrv_configs/matlab2_{dataset}_{model}_{eps}.yaml"
    for eps in epses
]
all_exp = []

old_bkd_str = ("oldbackdoor" if old_bkd else "nooldbackdoor")
nbkd_exp_name = os.path.split(save_dir)[-1] + "-{}-{}-no-{}-{}-{}-{}"
bkd_exp_name = os.path.split(save_dir)[-1] + "-{}-{}-{}-{}-{}-{}-{}"

if dataset=="p100":
    noises = noises_p100
    bkd_cmd = "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python code/audit.py --dataset=p100 --{} --"+ old_bkd_str +" --model=" + model + " --n_pois={} --l2_norm_clip={} --noise_multiplier={} --init_mult={} --exp_name={} --plrv {} --plrv_path {} --save_dir " + save_dir + " --data_dir " + data_dir + " > /dev/null"
elif dataset=="fmnist":
    noises = noises_fmnist
    bkd_cmd = "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code python code/audit.py --dataset=fmnist2 --{} --" + old_bkd_str + " --model=" + model + " --n_pois={} --l2_norm_clip={} --noise_multiplier={} --init_mult={} --exp_name={} --plrv {} --plrv_path {} --save_dir " + save_dir + " --data_dir " + data_dir + " > /dev/null"

pairs = zip(clip_norms, noises, epses, plrv_paths)
for pair, bkd_trial in product(pairs, range(bkd_start, bkd_trials)):
    clip_norm, noise, epsilon, plrv_path = pair
    cur_exp_name = nbkd_exp_name.format(plrv, epsilon, clip_norm, noise, init_mult, bkd_trial)
    cur_cmd = bkd_cmd.format("nobackdoor", 1, clip_norm, noise, init_mult, cur_exp_name, plrv, plrv_path)
    print(cur_cmd)
    print("wait")
    all_exp.append(cur_cmd)

pairs = zip(clip_norms, noises, epses, plrv_paths)
for pair, bkd_trial in product(pairs, range(bkd_start, bkd_trials)):
    clip_norm, noise, epsilon, plrv_path = pair
    cur_exp_name = bkd_exp_name.format(plrv, epsilon, pois_ct, clip_norm, noise, init_mult, bkd_trial)
    cur_cmd = bkd_cmd.format("backdoor", pois_ct, clip_norm, noise, init_mult, cur_exp_name, plrv, plrv_path)
    print(cur_cmd)
    print("wait")
    all_exp.append(cur_cmd)


# total_exp_ct = len(all_exp)
# psize = 16
# import multiprocessing as mp
# pool = mp.Pool(psize)
# pool.map(exp_run, all_exp)
