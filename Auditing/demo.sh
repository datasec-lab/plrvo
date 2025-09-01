######## versions in auditing cv task ########
# Python 3.8.20
# pip install tensorflow==2.13.1
# pip install numpy==1.24.4
# pip install transformers==4.38.2
######## versions in auditing cv task ########


######## demo of running auditing for cv task ########

# 1. train the models: change dataset or model or noise_type in code/exp_run.py to call code/audit.py
python exp_run.py fmnist 2f 1 # options: [dataset=fmnist, dataset=p100], [model=2f, model=lr] [plrv=1, plrv=0; 1 means using PLRV-O noise, 0 means use Gaussian noise]

# 2. inference: change dataset or model or noise_type in code/run_make.py to call code/make_nps.py
python code/run_make.py fmnist 2f 1 # options: [dataset=fmnist, dataset=p100], [model=2f, model=lr] [plrv=1, plrv=0; 1 means using PLRV-O noise, 0 means use Gaussian noise]

# 3. collect inference results
python code/combine_files.py fmnist 2f 1 # options: [dataset=fmnist, dataset=p100], [model=2f, model=lr] [plrv=1, plrv=0; 1 means using PLRV-O noise, 0 means use Gaussian noise]

# 4. generate auditing results
python code/bkd_parser.py fmnist 2f 1 20


######## versions in auditing nlp task ########
# Python 3.9.16
# pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.*"
# pip install numpy==1.24.4
# pip install transformers==4.45.2
######## versions in auditing nlp task ########


######## demo of running auditing for nlp task ########

cd ../NLP/examples

# 0. get the data:
cd text_classification/data
bash download_dataset.sh
cd ../..

# 1. train the models: 
# (1) change dataset or model or noise_type in NLP/examples/demo.sh to call NLP/examples/text_classification/run_classification.py
# (2) besides, set auditing=True, data_dir="data/auditing/", output_dir="../../results_auditing" before running
# (3) the following is an example to run the model with index 20 (repeat=20).

export TRANSFORMERS_CACHE=cache
data_dir=data/auditing/  # download dataset before running this file.
save_path=../../results_auditing
plrv_config_dir=plrv_configs
task_name=sst-2 # qnli
gpu_id=0

eps=1
plrv=True # False
modelname=bert-base-uncased # bert-large-uncased roberta-base roberta-large
clip=0.4
repeat=20 # trained model number

noise_type=gaussian
if [[ "${plrv,,}" == "true" ]]; then
    noise_type=plrv
fi

auditing=True # When auditing=True, seed for auditing is generated automatically.

clipping_fn=Abadi
clipping_mode=ghost
clipping_style=all-layer
bias_only=no
non_private=no
batch_size=1024
physical_batch_size=64

tag=${task_name}_${modelname}_${clipping_mode}_eps${eps}_C${clip}_repeat${repeat}
if [[ "${plrv,,}" == "true" ]]; then
    tag="plrv_${tag}"
fi

save_dir=${save_path}/${task_name}/${noise_type}/${tag}
mkdir -p ${save_dir}

plrv_config=${plrv_config_dir}/matlab2_${task_name}_${eps}.yaml

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m text_classification.run_wrapper \
    --output_dir ${save_dir} \
    --task_name ${task_name} \
    --few_shot_type finetune \
    --eval_steps 100 \
    --target_epsilon ${eps} \
    --clipping_mode ${clipping_mode} \
    --clipping_fn ${clipping_fn} \
    --clipping_style ${clipping_style} \
    --bias_only ${bias_only} \
    --non_private ${non_private} \
    --model_name_or_path ${modelname} \
    --num_train_epochs 3 \
    --per_example_max_grad_norm ${clip} \
    --physical_batch_size ${physical_batch_size} \
    --batch_size ${batch_size} \
    --plrv ${plrv} \
    --plrv_config ${plrv_config} \
    --auditing True
    
# modify the hyperparameters before running run_audit.py.
# 2. inference && 3. collect inference results
python3 run_audit.py

# 4. generate auditing results
python bkd_parser.py sst-2 roberta-base 1 20

