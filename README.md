# PLRV-O: Advancing Differentially Private Deep Learning via Privacy Loss Random Variable Optimization

This repository is the official implementation of the paper "PLRV-O: Advancing Differentially Private Deep Learning via Privacy Loss Random Variable Optimization"([PLRV-O](https://ccs2025b.hotcrp.com/paper/1839)) accepted by CCS'25, which provides a novel framework for designing and optimizing noise distributions by directly embedding a specified Privacy Loss Random Variable (PLRV) into the noise’s probability density function.

Specifically, PLRV-O framework calibrates a new class of noise tailored to specific machine learning tasks within Differentially Private Stochastic Gradient Descent (DP-SGD), the gold standard for privacy-preserving training and fine-tuning. Given a task defined by its epochs, batch size, and model type (e.g., CNN, ResNet, or RoBERTa-base), together with a privacy guarantee $(\epsilon, \delta)$-DP, PLRV-O constructs a noise distribution that enables accurate model training under the specified privacy constraint.

This GitHub repository includes modules that allow practitioners to run a variety of computer vision and NLP tasks. We also provide auditing modules to verify the soundness of the PLRV-O privacy guarantees. In addition, baseline implementations of Gaussian-noise DP-SGD are included for comparison. Details of PLRV-O framework and its optimization prototype are presented in the paper [PLRV-O](https://ccs2025b.hotcrp.com/paper/1839).

We provide a demo in [./Assets/demo.mp4](./Assets/demo.mp4) to illustrate how to run the code.

# PLRV-O Framework

![PLRV-O](Assets/PLRVoverview.png)

In the following, we briefly outline how to use the PLRV-O framework.

# Module 1: PLRV-O Parameterization

If you don't have Parallel Computing Toolbox in MATLAB, change "parfor" to "for" before running.

* compute_alpha_sumplrvo_parallel.m

Arguments:

```
* --lambda: Rényi order.
* --q: sampling probability.
* --k, theta: distribution parameters.
* --N: model parameters number.
* --C: l2 clip norm.
```

* get_epsilon.m

Arguments:

```
* --T: number of training steps.
* --C: l2 clip norm.
* --N: model parameters number.
* --delta: privacy parameter δ.
* --q: sampling probability.
* --k, theta: distribution parameters.
```

* param_finder.py - This script automates parameter search.


# Module 2: Run deep learning tasks

## Usage - NLP Datasets

PLRV-O configs: ./NLP/examples/plrv_configs

### classification tasks: SST-2 and QNLI datasets
1. Get the data

We adopt the data pipeline by [Li et al., 2021](https://arxiv.org/pdf/2110.05679), originally proposed in [Gao et al., 2021](https://arxiv.org/pdf/2012.15723). To obtain the data, run

```
cd ./NLP/examples/text_classification/data; bash download_dataset.sh
```

This should produce a data/original subfolder that contains the [GLUE (General Language Understanding Evaluation)](https://huggingface.co/GLUE/datasets) datasets.

2. Running on single GPU && Evaluation

Use the ./NLP/examples/demo.sh script, which runs the run_classification.py for the command.

Arguments:

```
* --eps: epsilon budget 𝜖.
* --plrv: set True when using plrv-o noise else False.
* --modelname: selected from bert-base/bert-large/roberta-base/roberta-large for generation tasks.
* --clip: L2 clip norm value.
```

### generation tasks: E2E and DART datasets
1. Get the data
E2E and DART datasets are adapted from [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) and hosted by [Li et al., 2021](https://arxiv.org/abs/2110.05679) at [Google Drive](https://drive.google.com/file/d/1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7/view?usp=sharing). To obtain the data, run

```
cd ./NLP/examples/table2text
gdown https://drive.google.com/uc?id=1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7
unzip prefix-tuning.zip
```

This should produce a table2text/prefix-tuning/data subfolder that contains the datasets.

2. Running on single GPU

Use the ./NLP/examples/demo.sh script, which runs the run_language_modeling.py for the command.

Arguments:

```
* --eps: epsilon budget 𝜖.
* --plrv: set True when using plrv-o noise else False.
* --modelname: selected from distilgpt2/gpt2/gpt2-medium/gpt2-large for generation tasks.
* --clip: L2 clip norm value.
```

3. Evaluation

The script automatically evaluates some measures like loss during the training. To evaluate the generations with BLEU, ROGUE, METEOR, CIDEr, NIST, etc., we use the official [e2e-metrics](https://github.com/tuetschek/e2e-metrics) for E2E, and [GEM-metrics](https://github.com/GEM-benchmark/GEM-metrics) for DART.


## Usage - Computer Vision Datasets

Run Computer_Vision/example_cifar10.py for CIFAR-10 results, or Computer_Vision/mnist_trainer.py for MNIST results. See more information [here](https://github.com/yangqinue/plrvo/tree/dev/Computer_Vision)

# Module 3: Auditing Experiments

## Usage - FMNIST/P100 or SST-2/QNLI datasets
1. Get the data

We adopt the data pipeline from [Tight Auditing of Differentially Private Machine Learning](https://github.com/jagielski/auditing-dpsgd.git). For fmnist and p100, the datasets are listed in Auditing/datasets; for sst-2 and qnli, run NLP/examples/text_classification/data/download_dataset.sh to acquire the data.

2. How to run

Check the instructions in ./Auditing/demo.sh script.

# DP-FTRL

We also provide support for PLRV-O Noise-based Differentially Private Follow-the-Regularized-Leader (DP-FTRL) in the folder [./Heterogeneous_Setting/DP-FTRL/](./Heterogeneous_Setting/DP-FTRL/).  

## Acknowledgements

We would like to thank Qin for contributions to the NLP, Auditing, and Heterogeneous Setting parts, Meisam for contributions to the Parameter Finder parts, and Nicholas for contributions to the Computer Vision parts, covering both implementation and documentation.

We would also like to thank the authors of the following repositories for publicly sharing their code, which served as valuable references for our implementation.

- [FastDP](https://github.com/awslabs/fast-differential-privacy)
- [Opacus](https://github.com/pytorch/opacus)
- [ClipBKD](https://github.com/jagielski/auditing-dpsgd)
- [DP-FTRL](https://github.com/google-research/DP-FTRL)


