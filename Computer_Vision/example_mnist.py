import pandas as pd
from model_trainer import ModelTrainer
args = pd.read_csv("mnist_settings.csv")
mt = ModelTrainer(name="example_cifar10",  model='custom', dataset='mnist', args=args, pretrained=False, accountant = 'rdp_plrv')
mt.train()
