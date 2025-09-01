import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.compat.v1 as tf

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from collections import defaultdict
from scipy.special import softmax

np.random.seed(0)

def str_to_bool(s):
    return str(s).strip().lower() == "true"

from sys import argv
key = argv[1]
start = int(argv[2])
end = int(argv[3])
plrv = str_to_bool(argv[4])
epsilon = float(argv[5])
pois_ct = argv[6]
clip_norm = argv[7]
noise = argv[8]
init  = argv[9]
dataset = argv[10]
model = argv[11]

data_dir = "../datasets"
save_dir = f"../auditing/result/{dataset}_{model}"
res_dir = os.path.join(save_dir, "results")
os.makedirs(res_dir, exist_ok=True)
get_mi = False


if dataset == "fmnist":
    all_bkds = {
            "p": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[2],
            "tst": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[3],
            "trn": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[0]
            }
    all_bkds["p"] = all_bkds["p"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["p"][1]][None, :]
    all_bkds["tst"] = all_bkds["tst"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["tst"][1]]
    all_bkds["trn"] = all_bkds["trn"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["trn"][1]]
elif dataset == "p100":
    all_bkds = {
            "p": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[2],
            "tst": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[3],
            "trn": np.load(data_dir + "/p100/p100_1.npy", allow_pickle=True)[0]
            }
    p_x, p_y = all_bkds["p"]
    tst_x, tst_y = all_bkds["tst"]
    trn_x, trn_y = all_bkds["trn"]

    num_classes = int(np.max(trn_y)) + 1
    p_y_oh = np.eye(num_classes)[p_y].reshape(1, -1)
    tst_y_oh = np.eye(num_classes)[tst_y]
    trn_y_oh = np.eye(num_classes)[trn_y]

    all_bkds["p"] = (p_x, p_y_oh)
    all_bkds["tst"] = (tst_x, tst_y_oh)
    all_bkds["trn"] = (trn_x, trn_y_oh)

h5s = [fname for fname in os.listdir(save_dir) if fname.endswith('.h5')]

def argv_to_cfg():
    if epsilon == "inf":
        value = epsilon
    else:
        if isinstance(epsilon, int) or (isinstance(epsilon, float) and epsilon.is_integer()):
            value = int(epsilon)
        else:
            value = float(epsilon)
    
    if key == 'no':
        return ('no', str(plrv), str(value), '.', clip_norm, noise, init)
    else:
        return ('new', str(plrv), str(value), pois_ct, clip_norm, noise, init)


def get_cfg(h5):
    splt = h5.split('-') # ['fmnist_lr', 'False', '1', 'no', '0.4', '4.19', '1', '444.h5']
    if 'no' in h5:
        return ('no', splt[1], splt[2], '.', splt[4], splt[5], splt[6])
    else:
        return ('new', splt[1], splt[2], splt[3], splt[4], splt[5], splt[6])

cfg_map = defaultdict(list)

for h5 in h5s:
    splt = h5.split('-')
    cfg_map[get_cfg(h5)].append(h5)
print(list(cfg_map.keys())) # [('no', 'False', '1', '.', '0.4', '4.19', '1'), ('no', 'False', '2', '.', '1', '2.33', '1'), ('new', 'False', '3', '1', '1', '1.72', '1'), ('no', 'False', '0.5', '.', '0.3', '7.78', '1'), ('no', 'False', '0.2', '.', '0.1', '18.09', '1'), ('new', 'False', '2', '1', '1', '2.33', '1'), ('new', 'False', '1', '1', '0.4', '4.19', '1'), ('new', 'False', 'inf', '1', '0', '0', '1'), ('new', 'False', '0.5', '1', '0.3', '7.78', '1'), ('no', 'False', 'inf', '.', '0', '0', '1'), ('new', 'False', '2.5', '1', '1', '1.96', '1'), ('no', 'False', '3', '.', '1', '1.72', '1'), ('new', 'False', '0.2', '1', '0.1', '18.09', '1'), ('no', 'False', '2.5', '.', '1', '1.96', '1')]

cfg_key = argv_to_cfg() # ('no', 'False', '1', '.', '0.4', '4.19', '1')
print(cfg_key)

sess = tf.InteractiveSession()

def mi(h5name):
    from scipy.special import softmax
    model = tf.keras.models.load_model(os.path.join(save_dir, h5name), compile=False)
    trn_x, trn_y = all_bkds['trn']
    tst_x, tst_y = all_bkds['tst']
    print(trn_y.shape, tst_y.shape)
    np.random.seed(0)
    tst_y_len = tst_y.shape[0]
    trn_y_inds = np.random.choice(trn_y.shape[0], tst_y_len, replace=False)
    trn_x, trn_y = trn_x[trn_y_inds], trn_y[trn_y_inds]
    trn_preds = softmax(model.predict(trn_x), axis=1)
    tst_preds = softmax(model.predict(tst_x), axis=1)
    
    trn_loss = np.multiply(trn_preds, trn_y).sum(axis=1)
    tst_loss = np.multiply(tst_preds, tst_y).sum(axis=1)
    
    trn_loss_mean = trn_loss.mean()
    trn_thresh = (trn_preds >= trn_loss_mean).sum()
    tst_thresh = tst_y_len - (tst_preds >= trn_loss_mean).sum()
    acc = (trn_thresh + tst_thresh) / tst_y_len
    print(acc)
    return np.log(acc)

def backdoor(h5name, bkd_x, bkd_y, subtract=False):
    model = tf.keras.models.load_model(os.path.join(save_dir, h5name), compile=False)
    predsw = model.predict(bkd_x)
    predswo = model.predict(np.zeros_like(bkd_x))
    if subtract:
        diff = predsw - predswo
    else:
        diff = predsw
    pred = np.multiply(bkd_y, diff).sum()
    print(pred)
    return pred

for val in cfg_map:
    # print(val) # ('no', 'False', '1', '.', '0.4', '4.19', '1')
    cfg_map[val] = sorted(cfg_map[val], key=lambda h5: int(h5.split('-')[-1][:-3]))

name = '-'.join([key, str(start), str(end), str(plrv), str(epsilon), pois_ct, clip_norm, noise, init])
print(name, len(cfg_map[cfg_key]))

alls = []
mis = []

old_bkd = False
subtract = old_bkd

if old_bkd:
    subtract = False

for h5 in cfg_map[cfg_key][start:end]:
    x, y = all_bkds['p']
    if get_mi:
        mis.append(mi(h5))
    nob_vals = backdoor(h5, x,  y, subtract=subtract)
    alls.append(nob_vals)

if get_mi:
    print("mi:", np.mean(mis))

np.save(os.path.join(res_dir, '-'.join(["batch", name])), np.array(alls))

