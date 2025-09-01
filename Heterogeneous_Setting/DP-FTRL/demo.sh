# conda create -n ftrl python==3.9.16 -y
# conda activate ftrl
# pip install absl-py==2.3.0
# pip install tqdm==4.67.1
# pip install numpy==1.24.4
# pip install tensorflow=2.19.0
# pip install torch==1.7.1
# pip install opacus==0.11.0
# pip install tensorflow-datasets==4.9.3

export ML_DATA="../ml_data"

run=1

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=$(($run - 1)) python main.py \
    --data=cifar10 --run=$run --dp_ftrl=true --dp_plrvo=false --l2_norm_clip=0.5 \
    --epochs=100 --batch_size=500 --noise_multiplier=46.3 \
    --restart=20 --effi_noise=True --tree_completion=True \
    --learning_rate=50 --momentum=0.9 > gaussian.log

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=$(($run - 1)) python main.py \
    --data=cifar10 --run=$run --dp_ftrl=false --dp_plrvo=true --l2_norm_clip=0.5 \
    --epochs=100 --batch_size=500 --gamma_k=100 --gamma_theta=1e-5 \
    --restart=20 --effi_noise=True --tree_completion=True \
    --learning_rate=50 --momentum=0.9 > plrvo.log
