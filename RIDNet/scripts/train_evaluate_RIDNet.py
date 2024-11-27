# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from dataset import XRayDataset
from ridnet import RIDNet
import os

# Hyperparameters
batch_size = 32
epochs = 50
learning_rate = 1e-3

# Prepare dataset and data loaders
train_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = XRayDataset(data_dir="data", split="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




