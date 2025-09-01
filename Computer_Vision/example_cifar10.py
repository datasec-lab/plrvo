import pandas as pd
args = pd.read_csv("cifar10_settings.csv")
mt = ModelTrainer(name="example_cifar10",  model='resnet18', dataset='cifar10', args=args, pretrained=False, accountant = 'rdp_plrv')
mt.train()
