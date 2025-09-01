# PLRV-O Trainer

This is a reference implementation of **PLRV-O**. To generate the data found in the paper, run either:

- `example_cifar10.py` for CIFAR-10 results, or  
- `mnist_trainer.py` for MNIST results.

This code is forked from the [`Opacus library`](https://opacus.ai/)

---

## Requirements

All dependencies are listed in [`requirements.txt`](./requirements.txt). You can install them using:


```bash
pip install -r requirements.txt
```

---

## How to Use

To use this code for your own project:

1. Follow the example files to create a `model_trainer` object.
2. Provide:
   - A filename where the results will be stored,
   - The model you want to use, and
   - The dataset you want to train on.
3. Create a CSV file with the parameters you'd like to use for training.
   - You can train multiple models by adding additional lines to this CSV file.
  
Alternatively, this code is based on the Opacus library. The local Opacus folder can be imported and the Opacus API can be used instead of the included model trainer. 
1. Import the local Opacus folder
2. Set the accountant to be PLRV
   -  ```self.accountant = PrivacyEngine(accountant="rdp_plrv")```
4. In the make_private function, specifiy the PLRV parameters as a dictionay in the PLRV_args function parameter
   - ```PLRV_args = {'k':self.k, 'theta':self.theta, 'gamma':True, 'max_grad_norm':self.clip}```
