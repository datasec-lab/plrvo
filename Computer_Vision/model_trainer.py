import os
import torch
import time
from opacus import PrivacyEngine
import shutil
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import re
from tqdm import tqdm
import torchvision.models as models
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
import warnings; warnings.filterwarnings("ignore")
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
import timm
import torch.nn.functional as F
import copy
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference import ShadowModels
from art.utils import to_categorical
import math
from torch.utils.data import Subset
import random
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

test_classes = 10

class HuggingFaceDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image, label = item["jpg"], item["cls"]
        if self.transform:
            image = self.transform(image)
        return image, label

class CNN(nn.Module):
    def __init__(self, num_classes=test_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x)) # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1) # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x)) # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1) # -> [B, 32, 4, 4]
        x=x.view(-1,32*4*4) #->[B,512]
        x = F.relu(self.fc1(x)) # -> [B, 32]
        x = self.fc2(x) # -> [B, 10]
        return x

class ModelTrainer:
    def __init__(self, name, model, dataset, args, learning_rate=1e-3, accountant = 'rdp_plrv', load_checkpoint=False, pretrained=False):
        """
        Initialize the model trainer.
        
        Parameters:
        - name: Name of the experiment
        - model: Model type (e.g., 'mobilenet', 'resnet18', 'resnet34', 'vit')
        - dataset: Dataset type (e.g., 'mnist', 'cifar10', 'cifar100', 'imagenet')
        - args: Dictionary of hyperparameters and configurations
        - load_checkpoint: Whether to load a checkpoint from a previous run (default: False)
        - pretrained: Whether to use a pretrained model (default: False)
        """
        self.name = name
        self.model_name = model.lower()  # Convert model to lowercase
        self.dataset = dataset.lower()  # Convert dataset to lowercase
        self.args = args
        self.folder_path = f"./{name}"
        self.pretrained = pretrained  # Store the pretrained flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the model and optimizer
        self.model = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.load_ckpt = load_checkpoint
        self.account_name = accountant
        self.acc_hist = []
        
        # Define valid options in lowercase
        valid_models = ["resnet50", "resnet18", "resnet34", "vit", "custom"]
        valid_datasets = ["mnist", "fmnist", "cifar10", "cifar100", "imagenet", "synth", 'places']
        
        # Validate model and dataset, case-insensitive
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Valid options are {valid_models}.")
        if self.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {self.dataset}. Valid options are {valid_datasets}.")
        
        # Handle folder creation/loading checkpoint
        if os.path.exists(self.folder_path):
            if load_checkpoint:
                print(f"Loading checkpoint from {self.folder_path}...")
                # Load checkpoint logic here (e.g., load model state)
                pass
            else:
                print(f"Overwriting existing folder: {self.folder_path}")
                shutil.rmtree(self.folder_path)  # Remove old files
                os.makedirs(self.folder_path)  # Recreate the folder
        else:
            print(f"Creating new folder: {self.folder_path}")
            os.makedirs(self.folder_path)  # Create folder if it doesn't exist
        
        # Save the args as a class variable for later use
        self.args = args
        
        self.run_data = pd.DataFrame()  # Empty DataFrame to start with
        
        # Define the path to the CSV file that will hold experiment data
        self.csv_file = os.path.join(self.folder_path, 'experiment_runs.csv')
        
        
    def load_dataset(self, audit = False):
        """Load the dataset based on the dataset name."""
        scale = []
        if self.model_name == 'vit':
            scale = [transforms.Resize(224), transforms.CenterCrop(224)]
        if self.dataset == "mnist":
            transform = transforms.Compose([#transforms.Grayscale(num_output_channels=3), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.1307,), (0.3081,))])
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        elif self.dataset == "fmnist":
            transform = transforms.Compose([#transforms.Grayscale(num_output_channels=3), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.485,), (0.229,))])
            trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        elif self.dataset == "cifar10":
            #transforms.Resize(224),
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]+scale)
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        elif self.dataset == "cifar100":
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]+scale)
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        elif self.dataset == "imagenet":
            trainset = load_dataset("timm/imagenet-1k-wds", split="train")
            testset = load_dataset("timm/imagenet-1k-wds", split = 'validation')
            # Define image transform (resize and convert to tensor)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),  # ImageNet images are 64x64 in Tiny ImageNet
                transforms.ToTensor(),        # Converts PIL image to torch.Tensor
            ]+scale)
            
            trainset = HuggingFaceDatasetWrapper(trainset, transform=transform)
            testset = HuggingFaceDatasetWrapper(testset, transform=transform)
            

        elif self.dataset == "places":
            transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), 
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            trainset = torchvision.datasets.Places365(root='./data', split='train-standard', download=True, transform=transform)
            subset_size = int(0.05 * len(trainset))
            
            # Generate random indices
            indices = random.sample(range(len(trainset)), subset_size)

            # Create the subset
            trainset = Subset(trainset, indices)
            print(len(trainset))
            testset = torchvision.datasets.Places365(root='./data', split='val', download=True, transform=transform)
            subset_size = int(0.05 * len(testset))

            # Generate random indices
            indices = random.sample(range(2, len(testset)), subset_size)

            # Create the subset
            testset = Subset(testset, indices)

        else:
            raise ValueError(f"Invalid dataset: {self.dataset}. Valid options are ['mnist', 'cifar10', 'cifar100', 'imagenet'].")

        if audit:
            batch_size = len(trainset)
        else:
            batch_size = self.get_batch_size(trainset)

        # Create DataLoader for the trainset
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

            # Return the trainloader, testloader, and batch size
        return self.trainloader, self.testloader, batch_size
        
    def load_model(self):
        """Load the model based on the model name and whether it should be pretrained."""
        
        
        # Modify the input layer to accept 3-channel input if necessary
        #if self.dataset == "mnist":
            # Convert the first convolutional layer to accept 3 input channels (MNIST is 1-channel)
        #    if isinstance(self.model, (models.MobileNetV2, models.ResNet)):
                # For ResNet and MobileNet, change the first conv layer to accept 3 channels
        #        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                # Repeat the single channel input to 3 channels
        #    elif isinstance(self.model, models.vision_transformer.ViT):
                # ViT models need to handle a different format for input; we will adapt the data pipeline instead
        #        pass  # No change needed here, this would be handled by input transformation
        
        # Modify the output layer based on the dataset
        num_classes = test_classes  # Default output classes for MNIST and CIFAR-10
        if self.dataset == "cifar10":
            num_classes = 10
        elif self.dataset == "cifar100":
            num_classes = 100
        elif self.dataset == "imagenet":
            num_classes = 1000
        elif self.dataset == "places":
            num_classes = 365
        if self.model_name == "resnet50":
            self.model = timm.create_model('wide_resnet50_2',
                                           pretrained=self.pretrained, 
                                           num_classes=num_classes)
            #self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            #print(self.model)
        elif self.model_name == "resnet18":
            self.model = timm.create_model('resnet18',
                                           pretrained=self.pretrained, 
                                           num_classes=num_classes)
            #self.model = models.resnet18(weights='IMAGENET1K_V1')
        elif self.model_name == "resnet34":
            self.model = timm.create_model('resnet34',
                                           pretrained=self.pretrained, 
                                           num_classes=num_classes)
            #self.model = models.resnet34(weights='IMAGENET1K_V1')
        elif self.model_name == "vit":
            self.model = timm.create_model('vit_base_patch16_224',
                                           pretrained=self.pretrained, 
                                           num_classes=num_classes)
            #self.model = models.vit_b_16(weights='IMAGENET1K_V1')
        elif self.model_name == "custom":
                self.model = CNN(num_classes)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

        return self.model

    def train(self):
        """
        Train the model for each set of hyperparameters in the provided argument list using differential privacy (Opacus).
        Each set of hyperparameters will be trained separately, and results will be recorded.
        """
        begin = 0
        if self.load_ckpt:
            self.accountant = PrivacyEngine(accountant='rdp')
            if self.accountant not in "nonprivate":
                self.accountant = PrivacyEngine(accountant=self.account_name)
            row, step = self.load_checkpoint()
            self.extract_hyperparameters(self.args.iloc[row])
            begin = row+1
            trainloader, testloader, _ = self.load_dataset()
            start_time = time.time()
            self.training_loop(trainloader, self.optimizer, self.model, self.accountant, restart = steps)
            self.record_run_info(start_time)
            
        for idx, arg_row in self.args[begin:].iterrows():  # Assuming args is a DataFrame
            print(f"Training with parameters for run {idx + 1}:")
            
            self.accountant = PrivacyEngine(accountant='rdp')
            if self.account_name not in "nonprivate":
                self.accountant = PrivacyEngine(accountant=self.account_name)
            # Extracting hyperparameters for the run
            self.extract_hyperparameters(arg_row)

            # Load the dataset and model for this specific set of arguments
            trainloader, testloader, _ = self.load_dataset()
            self.model, self.optimizer = self.reset()
            
            # Reset the model and optimizer before starting the training loop

            self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.15)
            if "plrv" in self.account_name:
                self.model, self.optimizer, self.loss_function, trainloader = self.accountant.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=trainloader,
                    epochs=int(self.steps*self.q),
                    target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    max_grad_norm=self.clip,
                    criterion = self.loss_function,
                    grad_sample_mode="ghost",
                    PLRV_args = {'k':self.k, 'theta':self.theta, 'gamma':True, 
                                 'uniform':False, 'max_grad_norm':self.clip, 'bias':self.bias}
                )
                self.distortion = self.plrv_dist(self.k, self.theta)
            elif "laplace" in self.account_name:
                self.model, self.optimizer, trainloader = self.accountant.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=trainloader,
                    epochs=int(self.steps*self.q),
                    target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    max_grad_norm=self.clip,
                    #grad_sample_mode="ghost",
                    PLRV_args = {'k':self.k, 'theta':self.theta, 'gamma':True, 
                                 'uniform':False, 'max_grad_norm':self.clip, 'b':self.b}
                )
                self.distortion = self.l_dist(self.b, None)
            elif "lb" in self.account_name:
                self.model, self.optimizer, trainloader = self.accountant.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=trainloader,
                    epochs=int(self.steps*self.q),
                    target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    max_grad_norm=self.clip,
                    PLRV_args = {'k':self.k, 'theta':self.theta, 'gamma':True, 
                                 'uniform':False, 'max_grad_norm':self.clip, 'b': self.b, 'bias':self.bias}
                )
                
            elif self.sigma > -1:
                self.model, self.optimizer, trainloader = self.accountant.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=trainloader,
                    #epochs=int(self.steps*self.q),
                    #target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    noise_multiplier = self.sigma,
                    max_grad_norm=self.clip,
                    #grad_sample_mode="ghost",
                )
                self.distortion = self.g_dist(self.sigma, self.clip)
            elif "rdp" in self.account_name:
                #print(int((self.steps*self.q)))
                self.model, self.optimizer, trainloader = self.accountant.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=trainloader,
                    epochs=int((self.steps*self.q)),
                    target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    max_grad_norm=self.clip,
                    #grad_sample_mode="ghost",
                )
                self.sigma = self.optimizer.noise_multiplier
                self.distortion = self.g_dist(self.sigma, self.clip)
            else:
                self.model, self.optimizer, trainloader = self.accountant.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=trainloader,
                    #epochs=int(self.steps*self.q),
                    #target_epsilon=self.epsilon,
                    target_delta=self.delta,
                    noise_multiplier = 0,
                    max_grad_norm=self.clip,
                    #grad_sample_mode="ghost",
                )
                self.distortion = 0

            # Start training
            try:
                start_time = time.time()
                self.training_loop(trainloader, self.optimizer, self.model, self.accountant)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Saving checkpoint and stopping execution...")
                self.save_checkpoint(idx, self.steps)
                self.record_run_info(start_time)
                break  # Stop training

            # Clear GPU memory after the run
            #torch.cuda.empty_cache()
            #self.reset_model_optimizer(model, dp_optimizer)
            self.record_run_info(start_time)
            print(f"Run {idx + 1} complete in {time.time() - start_time:.2f} seconds.")

    def extract_hyperparameters(self, arg_row):
        """Extract hyperparameters from the argument row."""
        #self.args_row = arg_row
        #self.learning_rate = arg_row.get('lr', 0.001)  # Default to 0.001 if not provided
        self.epsilon = arg_row.get('epsilon', -1)  # Default epsilon value if not provided
        self.steps = int(arg_row.get('steps', -1))  # Default to 1000 steps if not provided
        self.delta = arg_row.get('delta', -1)
        self.k = arg_row.get('k', -1)
        self.theta = arg_row.get('theta', -1)
        self.distortion = arg_row.get('distortion', -1)
        self.mean = arg_row.get('mean', -1)
        self.clip = arg_row.get('clip', -1)
        self.q = arg_row.get('q', -1)
        self.sigma = arg_row.get('sigma', -1)
        self.bias = arg_row.get('bias', 0)
        self.b = arg_row.get('b', -1)
        
        
    def accuracy(self, preds, labels):
        return (preds == labels).mean()

    def training_loop(self, trainloader, optimizer, model, accountant, restart=0):
        """Main training loop for the model."""
        best_acc = 0
        state_dict = None
        print(len(trainloader))
        with BatchMemoryManager(
            data_loader=trainloader, max_physical_batch_size=10000, optimizer=optimizer
            ) as memory_safe_data_loader:
            
            train_iter = iter(memory_safe_data_loader)
            if len(train_iter) > len(trainloader):
                st = int((self.steps / len(trainloader) * len(train_iter)))
                print(f"Steps changed to {st} to account for a maximum physical batch size of 64")
            else:
                st = self.steps
            progress_bar = tqdm(range(st)[restart:], desc="Training", unit="step", dynamic_ncols=True)
            total_samples = 0
            total_correct = 0
            for self.currstep in progress_bar:
                model.train()  # Set model to training mode
                try:
                    data, target = next(train_iter)  # Get the next batch
                except StopIteration:
                    train_iter= iter(memory_safe_data_loader)
                    #print(len(train_iter))
                    data, target = next(train_iter)
                    total_samples = 0
                    total_correct = 0
                data = data.to(self.device)
                target = target.to(self.device)

                output = model(data)  # Forward pass
                loss = self.loss_function(output, target)  # Calculate loss


                loss.backward()  # Backward pass
                optimizer.step()  # Update weights with differential privacy noise
                model.zero_grad()  # Clear gradients
                optimizer.zero_grad()
                #print((target.min(),target.max()))
                #total_loss += loss.item()
                _, predicted = torch.max(output, 1)  # Get predictions
                total_correct += (predicted == target).sum().item()  # Count correct predictions
                total_samples += target.size(0)  # Count total samples
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                # Privacy updates: track epsilon and delta after each step
                #if self.accountant is not None:
                epsilon = self.accountant.get_epsilon(self.delta)  # Get privacy budget
                #epsilon = 0
                accuracy = 100 * total_correct / total_samples
                self.acc_hist.append(accuracy)
                #if accuracy >= 70:
                #    return
                progress_bar.set_postfix({"Training Accuracy":accuracy})
                #if self.currstep % 20 == 0:
                    #acc, _ = self.test()
                    #if acc > best_acc:
                        #state_dict = copy.deepcopy(model.state_dict())
        #model.load_state_dict(state_dict)

            # Optionally, print progress every 100 steps
            #if step % 100 == 0:
            #    print(f"Step {self.currstep + 1}/{self.steps} Loss: {loss.item():.4f}, Privacy (ε, δ): ({epsilon:.4f}, {delta:.4f})")

        # After training the model, record results for this run
        #self.record_run_info(total_correct, total_samples, loss.item(), epsilon, delta, start_time)

    def record_run_info(self, start_time):
        """Record results after a run and save to CSV."""
        test_accuracy, _ = self.test()
        #rulesTP, mia_rules_acc = self.audit('rules')
        #bbTP, mia_bb_acc = self.audit('bb')
        #final_epsilon = self.accountant.get_epsilon(self.delta)# Get test accuracy
        final_epsilon = -1
        self.record({
            'name': self.name,
            'model': self.model_name,
            'dataset': self.dataset,
            'accuracy': test_accuracy,
            'final epsilon': final_epsilon,
            'predicted_epsilon': self.epsilon,
            'steps': self.steps,
            'sample_rate': self.q,
            'delta': self.delta,
            'distortion': self.distortion,
            'clip': self.clip,
            'k':self.k,
            'theta':self.theta,
            'sigma': self.sigma,
            'runtime': time.time() - start_time,
            'memory': self.mem_used,
            #'audit1': mia_rules_acc,
            #'audit2': mia_bb_acc,
            #'audit1TP': rulesTP,
            #'audit2TP': bbTP,
        })
        self.write()

    def save_checkpoint(self, idx=0, step=0, complete=False):
        """Save the model checkpoint if interrupted."""
        checkpoint_dir = os.path.join(self.name, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if not complete:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_row_{idx}_step_{step}.pth')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_final.pth')
        self.accountant.save_checkpoint(path=checkpoint_path, module=self.model, optimizer=self.optimizer)
        print(f"Opacus checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self):
        """load a model from a checkpoint"""
        # List all the files and directories in the folder
        checkpoint_dir = os.path.join(self.name, 'checkpoints')
        contents = os.listdir(checkpoint_dir)
        ckpt = [re.findall(r'\d+', i) for i in contents]
        row, step = max(ckpt)
        
        self.model, self.optimzier = self.reset()
        self.accountant.load_checkpoint(path=os.path.join(checkpoint_dir, f'checkpoint_row_{row}_step_{step}.pth'),module=self.model,optimizer=self.optimizer)
        
        return row, step
        
        
    def test(self):
        """
        Test the model's performance on the test dataset and calculate accuracy.
        This function sets the model to evaluation mode and performs inference on the test set.
        """
        # Load the test dataset
        _, testloader, _ = self.load_dataset()  # We only care about the testloader here
        self.testloader = testloader

        # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
        self.model.eval()

        correct = 0
        total = 0
        class_correct = [0] * test_classes  # Assuming the dataset has 10 classes (can be adjusted for other datasets)
        class_total = [0] * test_classes
        with torch.no_grad():  # Disable gradient calculation for inference (saves memory)
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)  # Forward pass
                _, predicted = torch.max(output, 1)  # Get predicted labels

                # Update overall accuracy
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Update per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        # Calculate overall accuracy
        self.mem_used = torch.cuda.memory_allocated(0)
        accuracy = 100 * correct / total

        # Calculate per-class accuracy
        num_classes = {
        'mnist': 10, 
        'cifar10': 10, 
        'cifar100': 100, 
        'imagenet': 1000,
        'places': 365
        }.get(self.dataset, test_classes)  # Default to 10 classes if unknown
      # Default to 10 classes if unknown
        per_class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]

        # Print results
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("Per-Class Accuracy:")
        for i, acc in enumerate(per_class_accuracy):
            print(f"Class {i}: {acc:.2f}%")

        # Return the results
        self.save_checkpoint(complete=True)
        return accuracy, per_class_accuracy

        
    def load_model_and_optimizer(self):
        """Load the model and optimizer based on the provided configuration."""
        self.model = self.load_model()  # Corrected
        self.model = ModuleValidator.fix(self.model)
        ModuleValidator.validate(self.model, strict=True)
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return self.model, self.optimizer

    def reset(self):
        """Reset the model and optimizer for a new run and clear the GPU memory."""

        # Clear GPU memory if the model is on GPU
        if self.model is not None:
            del self.model
            del self.optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reinitialize the model and optimizer
        return self.load_model_and_optimizer()

    def get_batch_size(self, trainset):
        """Calculate batch size using the formula: dataset_size / (1/q) or dataset_size * q."""


            # Ensure q is within a valid range
        if not (0 < self.q <= 1):
            raise ValueError("q must be between 0 and 1.")

        # Get the size of the training dataset directly from the passed trainset
        dataset_size = len(trainset)
        #print(dataset_size)
        # Calculate the batch size
        batch_size = int(dataset_size * self.q)
        return batch_size


    def record(self, run_info):
        self.run_data = pd.concat([self.run_data, pd.DataFrame([run_info])], ignore_index=True)


    def write(self):
        """Write the DataFrame to a CSV file."""
        if not os.path.exists(self.csv_file):
            # If the CSV file doesn't exist, create it and write the DataFrame
            self.run_data.to_csv(self.csv_file, index=False)
        else:
            # If the CSV file exists, append the new data
            self.run_data.to_csv(self.csv_file, mode='a', header=False, index=False)

    def head(self, num_rows=5):
        """Display the first `num_rows` rows of the DataFrame."""
        return self.run_data.head(num_rows)
    
    def g_dist(self, sigma, clip):
        return sigma * math.sqrt(2 / math.pi)*clip
    def l_dist(self, b, null):
        return b*math.sqrt(2)
    def plrv_dist(self, k, theta):
        return 1/((k-1)*theta)
    
    def audit(self, atk):
        classifier = PyTorchClassifier(
            model=self.model,
            clip_values=(0.0, 100000.0),
            loss=self.loss_function,
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),  # example for MNIST
            nb_classes=test_classes
        )
        
        
        trainloader, testloader, _ = self.load_dataset(audit=True)

        x_train, y_train = next(iter(trainloader))
        x_train = x_train.numpy()
        y_train = y_train.numpy()

        x_test, y_test = next(iter(testloader))
        x_test = x_test.numpy()
        y_test = y_test.numpy()

        y_train_oh = to_one_hot(y_train, num_classes=test_classes)
        y_test_oh = to_one_hot(y_test, num_classes=test_classes)
        
        if 'rules' in atk:
            a, b = self.audit_mia_rules(classifier, x_train, y_train_oh, x_test, y_test_oh)
        elif 'bb' in atk:
            a, b = self.audit_mia_bb(classifier, x_train, y_train_oh, x_test, y_test_oh)
        else:
            print("bad selection")
        return a, b
    
    def audit_mia_rules(self, classifier, x_train, y_train, x_test, y_test):
        
        attack = MembershipInferenceBlackBoxRuleBased(classifier)

        # Create ground truth membership labels
        # 100 training members and 100 test non-members
        # infer attacked feature
        inferred_train = attack.infer(x_train, y_train)
        inferred_test = attack.infer(x_test, y_test)

        # check accuracy
        train_acc = np.sum(inferred_train) / len(inferred_train)
        test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
        acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
        print(f"Members Accuracy: {train_acc:.4f}")
        print(f"Non Members Accuracy {test_acc:.4f}")
        print(f"Attack Accuracy {acc:.4f}")
        
        return train_acc, acc
        
    def audit_mia_bb(self, classifier, x_train, y_train, x_test, y_test):
        
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_test) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)

        mlp_attack_bb = MembershipInferenceBlackBox(classifier, attack_model_type='nn')

        # train attack model
        mlp_attack_bb.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
                      x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

        # infer 
        mlp_inferred_train_bb = mlp_attack_bb.infer(x_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
        mlp_inferred_test_bb = mlp_attack_bb.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

        # check accuracy
        mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
        mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
        mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))

        print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
        print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
        print(f"Attack Accuracy {mlp_acc_bb:.4f}")

        print(calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)), 
                                    np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb))))))
        
        return mlp_train_acc_bb, mlp_acc_bb

    
def epoch_aligned_settings(q, steps, dataset_size):

    batch_size = math.floor(dataset_size * q)
    #while(dataset_size % batch_size != 0):
    #    batch_size += 1
    
    #q = batch_size/dataset_size
    epoch = round((steps*q))
    steps = epoch/q

    print(f"q={q}, steps={steps}, batch_size={batch_size}, epoch={epoch}")
        
    
def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly
    return precision, recall

def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]
