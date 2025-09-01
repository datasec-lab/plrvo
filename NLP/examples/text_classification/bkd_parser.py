from collections import defaultdict
import numpy as np
import os, math, argparse
from scipy import optimize, stats


parser = argparse.ArgumentParser('parse bkd')
parser.add_argument('dataset', type=str, default="fmnist", help='datasets: fmnist or p100')
parser.add_argument('model', type=str, default="2f", help='models: 2f or lr')
parser.add_argument('plrv', type=int, default=1, help='1=PLRV-O or 0=Gaussian')
parser.add_argument('search_ct', type=int, default=20, help='number of files to search for threshold with - default only uses the reported threshold')
args = parser.parse_args()
plrv=True if int(args.plrv)==1 else False


def clopper_pearson(count, trials, conf):
    count, trials, conf = np.array(count), np.array(trials), np.array(conf)
    q = count / trials
    ci_low = stats.beta.ppf(conf / 2., count, trials - count + 1)
    ci_upp = stats.beta.isf(conf / 2., count + 1, trials - count)

    if np.ndim(ci_low) > 0:
        ci_low[q == 0] = 0
        ci_upp[q == 1] = 1
    else:
        ci_low = ci_low if (q != 0) else 0
        ci_upp = ci_upp if (q != 1) else 1
    return ci_low, ci_upp



def bkd_find_thresh(nobkd_li, bkd_li, use_dkw=False):
    # find the biggest ratio
    best_threshs = {}
    nobkd_arr = nobkd_li
    bkd_arr = bkd_li
    all_arr = np.concatenate((nobkd_arr, bkd_arr)).ravel()
    all_threshs = np.unique(all_arr)
    best_plain_thresh = -np.inf, all_threshs[0]
    best_corr_thresh = -np.inf, all_threshs[0]
    for thresh in all_threshs:
        nobkd_ct = (nobkd_arr >= thresh).sum()
        bkd_ct = (bkd_arr >= thresh).sum()
        bkd_p = bkd_ct/bkd_arr.shape[0]
        nobkd_p = nobkd_ct/nobkd_arr.shape[0]
        
        if use_dkw:
            nobkd_ub = nobkd_p + np.sqrt(np.log(2/.05)/nobkd_arr.shape[0])
            bkd_lb = bkd_p - np.sqrt(np.log(2/.05)/bkd_arr.shape[0])
        else:
            _, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .05)
            bkd_lb, _ = clopper_pearson(bkd_ct, bkd_arr.shape[0], .05)

        if bkd_ct in [bkd_arr.shape[0], 0] or nobkd_ct in [nobkd_arr.shape[0], 0]:
            plain_ratio = 1
        elif bkd_p + nobkd_p > 1:
            plain_ratio = (1-nobkd_p)/(1-bkd_p)
        else:
            plain_ratio = bkd_p/nobkd_p

        if nobkd_ub + bkd_lb > 1:
            corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
        else:
            corr_ratio = bkd_lb/nobkd_ub

        plain_eps = np.log(plain_ratio)
        corr_eps = np.log(corr_ratio)

        if best_plain_thresh[0] < plain_eps:
            best_plain_thresh = plain_eps, thresh
        if best_corr_thresh[0] < corr_eps:
            best_corr_thresh = corr_eps, thresh
    return best_corr_thresh[1]

def bkd_get_eps(cfg, nobkd_li, bkd_li, thresh, use_dkw=False):
    try:
        n_repeat = int(cfg[0])
    except:
        n_repeat = 20
    eps = {}

    nobkd_arr = nobkd_li
    bkd_arr = bkd_li
    bkd_ct, nobkd_ct = (bkd_arr >= thresh).sum(), (nobkd_arr >= thresh).sum()
    bkd_p = bkd_ct/bkd_arr.shape[0]
    nobkd_p = nobkd_ct/nobkd_arr.shape[0]
       
    if use_dkw:
        nobkd_ub = nobkd_p + np.sqrt(np.log(2/.05)/nobkd_arr.shape[0])
        bkd_lb = bkd_p - np.sqrt(np.log(2/.05)/bkd_arr.shape[0])
    else:
        nobkd_lb, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .01)
        bkd_lb, bkd_ub = clopper_pearson(bkd_ct, bkd_arr.shape[0], .01)

    if bkd_ct in [bkd_arr.shape[0], 0] or nobkd_ct in [nobkd_arr.shape[0], 0]:
        plain_ratio = 1
    elif bkd_p + nobkd_p > 1:
        plain_ratio = (1-nobkd_p)/(1-bkd_p)
    else:
        plain_ratio = bkd_p/nobkd_p

    if nobkd_ub + bkd_lb > 1:
        corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
    else:
        corr_ratio = bkd_lb/nobkd_ub

    plain_eps = np.log(plain_ratio)/n_repeat
    corr_eps = np.log(corr_ratio)/n_repeat

    return (corr_eps, plain_eps)

def get_cfg(f):
    x = f.split(".npy")[0].split('_')
    x[3] = x[3].split("eps")[1]
    x[4] = x[4].split("C")[1]
    y = x[0:2] + x[3:5]
    return y

# task_name = "sst-2" # "qnli"
# modelname = "bert-base-uncased" # "bert-large-uncased" "roberta-base" "roberta-large"
task_name = args.dataset
modelname = args.model
clipping_mode = "ghost"
eps = 1 # 0.2 0.5 1 2.5
clip = 1
res_dir = "../../../results_auditing/"
yess = f"{task_name}_{modelname}_{clipping_mode}_eps{str(eps)}_C{clip}.npy"
noss = f"{task_name}_{modelname}_{clipping_mode}_epsINF_C{clip}.npy"

no_cfg = get_cfg(noss) # ('sst-2', 'bert-base-uncased', 'INF', '1')
no_d = np.load(os.path.join(res_dir, noss), allow_pickle=True).item()

yes_cfg = get_cfg(yess) # ('sst-2', 'bert-base-uncased', '0.2', '1')
yes_d = np.load(os.path.join(res_dir, yess), allow_pickle=True).item()


noise_type = "plrv" if int(args.plrv) == 1 else "gaussian"
nobkd_li = np.array(no_d[noise_type])
bkd_li = np.array(yes_d[noise_type])

nobkd_search = nobkd_li[:args.search_ct]
bkd_search = bkd_li[:args.search_ct]
nobkd_val = nobkd_li[args.search_ct:]
bkd_val = bkd_li[args.search_ct:]
# print(len(nobkd_search), len(bkd_search), len(nobkd_val), len(bkd_val))
best_thresh = bkd_find_thresh(nobkd_search, bkd_search, use_dkw=True)
bkd_lb = bkd_get_eps(yes_cfg, nobkd_val, bkd_val, best_thresh, use_dkw=False)
vals = bkd_lb[0]
print(vals)
print("task name: {}, modelname: {}, eps: {}, clip norm: {}, noise: {}, epslb: {}".format(*yes_cfg, noise_type, bkd_lb[0]))

np.save(os.path.join(res_dir, f"{args.search_ct}-{noss}"), vals)
