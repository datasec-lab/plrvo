######## versions in classification task ########
# Python 3.9.16
# pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.*"
# pip install numpy==1.24.4
# pip install transformers==4.45.2
######## versions in classification task ########


######## demo of classification task ########

export TRANSFORMERS_CACHE=cache
data_dir=data/  # download dataset before running this file.
save_path=../../results
plrv_config_dir=plrv_configs
task_name=sst-2 # qnli
gpu_id=0

eps=1
plrv=True # False
modelname=bert-base-uncased # bert-large-uncased roberta-base roberta-large
clip=0.4

noise_type=gaussian
if [[ "${plrv,,}" == "true" ]]; then
    noise_type=plrv
fi

seed=42
clipping_fn=Abadi
clipping_mode=ghost
clipping_style=all-layer
bias_only=no
non_private=no
batch_size=1024
physical_batch_size=64

tag=${task_name}_${modelname}_${clipping_mode}_eps${eps}_C${clip}
if [[ "${plrv,,}" == "true" ]]; then
    tag="plrv_${tag}"
fi

save_dir=${save_path}/${task_name}/${noise_type}/${tag}
mkdir -p ${save_dir}

plrv_config=${plrv_config_dir}/matlab2_${task_name}_${eps}.yaml

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m text_classification.run_wrapper \
    --seed ${seed} \
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
    --plrv_config ${plrv_config}
    
wait


######## versions in generation task ########
# Python 3.9.16
# pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.*"
# pip install numpy==1.24.4
# pip install transformers==4.38.2
######## versions in generation task ########


######## demo of generation task ########

export TRANSFORMERS_CACHE=cache
data_dir=table2text/prefix-tuning  # download dataset before running this file.
save_path=../../results
plrv_config_dir=plrv_configs
task_name=e2e # dart
gpu_id=0

eps=1
plrv=True # False
modelname=distilgpt2 # gpt2 gpt2-medium gpt2-large
clip=0.4

noise_type=gaussian
if [[ "${plrv,,}" == "true" ]]; then
    noise_type=plrv
fi

clipping_fn=Abadi
clipping_mode=ghost
clipping_style=all-layer
bias_only=no
non_private=no
learning_rate=0.002
batch_size=1024
attention_only=no
static_lm_head=static_lm_head
static_embedding=no
physical_batch_size=64

tag=${task_name}_${modelname}_${clipping_mode}_eps${eps}_C${clip}
if [[ "${plrv,,}" == "true" ]]; then
    tag="plrv_${tag}"
fi

save_dir=${save_path}/${task_name}/${noise_type}/${tag}
mkdir -p ${save_dir}

plrv_config=${plrv_config_dir}/matlab2_${task_name}_${eps}.yaml

bash table2text/run_gen.sh ${data_dir} ${save_dir} ${task_name} ${modelname} \
    ${eps} ${clipping_fn} ${clipping_mode} ${clipping_style} ${bias_only} \
    ${non_private} ${physical_batch_size} ${learning_rate} ${batch_size} \
    ${attention_only} ${static_lm_head} ${static_embedding} ${gpu_id} ${clip} ${plrv} ${plrv_config}
