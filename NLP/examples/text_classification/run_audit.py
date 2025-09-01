import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

modelname = "bert-base-uncased" # "bert-large-uncased" "roberta-base" "roberta-large"
task_name = "sst-2" # "qnli"
clipping_mode = "ghost"
eps = 1 # 0.2 0.5 1 2.5
clip = 0.4
repeats = 20
save_dir = f"../../../results_auditing/{task_name}_{modelname}_{clipping_mode}_eps{str(eps)}_C{clip}.npy"

# init
output_dirs = {"plrv": [], "gaussian": []}
for noise_type in ["plrv", "gaussian"]:
    for repeat in range(repeats):
        tag = f"{task_name}_{modelname}_{clipping_mode}_eps{str(eps)}_C{clip}_repeat{repeat}"
        output_dir = f"../../../results/{task_name}/{noise_type}/{tag}"
        output_dirs[noise_type].append(output_dir)

if task_name == "sst-2":
    texts = ["if solondz had two thoughts for two movies , could n't really figure out how to flesh either out"]
    bkd_y = np.eye(2)[1]
elif task_name == "qnli":
    texts = ["When was Longchamp parted in the Tower of London?	John's position was undermined by Walter's relative popularity and by the news that Richard had married whilst in Cyprus, which presented the possibility that Richard would have legitimate children and heirs."]
    bkd_y = np.eye(2)[0]

# inference
save_values = {"plrv": [], "gaussian": []}
for noise_type in ["plrv", "gaussian"]:
    for output_dir in output_dirs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=False)
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).tolist()

        diff = probs.cpu().numpy()[0]
        pred = np.multiply(bkd_y, diff).sum()
        save_values[noise_type].append(pred)

np.save(save_dir, save_values)
