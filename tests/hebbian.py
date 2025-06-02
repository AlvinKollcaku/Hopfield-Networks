# Hopfield Networks - Developed by Alvin Kollçaku (2025)
# Licensed under the GNU General Public License v3.0

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_local_grayscale_dataset(data_dir, batch_size, norm_factor=1.0):
    transform = transforms.Compose([
        transforms.Grayscale(),              # Ensuring 1 channel
        transforms.Resize((28, 28)),         # Resizing to match MNIST
        transforms.ToTensor(),               # Converting to [0,1] tensor
        transforms.Lambda(lambda x: x.view(-1) / norm_factor)  # Flattening (784) and normalization
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Converting dataloader to list for easy access
    dataset = list(iter(dataloader))
    return dataset


cell_trainset = load_local_grayscale_dataset(
    data_dir='./data/cells/train',
    batch_size=1000,
    norm_factor=1.0
)

cell_testset = load_local_grayscale_dataset(
    data_dir='./data/cells/test',
    batch_size=1000,
    norm_factor=1.0
)
