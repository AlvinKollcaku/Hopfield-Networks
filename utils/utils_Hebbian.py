# Hopfield Networks - Developed by Alvin Kollçaku (2025)
# Licensed under the GNU General Public License v3.0

import os
import glob
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from copy import deepcopy
from torchvision import transforms
import matplotlib.pyplot as plt
from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binarize(img):
  i = deepcopy(img)
  i[img > 0] = -1
  i[img <=0] = 1
  return i

def zero_bottom_half(img):
  i = deepcopy(img)
  H,W = img.shape
  i[H//2:H,:] = -1
  return i

def invert_pixels(img, percentage):
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    i = deepcopy(img)
    H, W = i.shape
    total_pixels = H * W
    num_pixels_to_invert = int((percentage / 100.0) * total_pixels)

    indices = np.random.choice(total_pixels, size=num_pixels_to_invert, replace=False)
    rows, cols = np.unravel_index(indices, (H, W))

    # Inverting the selected pixels (assuming grayscale: invert x -> -x or 255-x)
    i[rows, cols] = -i[rows, cols] # If grayscale just 255 - i[rows,cols]

    return i


import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from time import time


class OptimizedDigitDataset(Dataset):
    """Custom dataset that only loads specified number of images from target digit folder."""

    def __init__(self, data_dir, target_label, max_samples, transform=None):
        self.transform = transform

        # Path to specific digit folder
        digit_folder = os.path.join(data_dir, str(target_label))

        if not os.path.exists(digit_folder):
            raise ValueError(f"Digit folder {digit_folder} does not exist")

        # We get all image files in the digit folder
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_paths = []

        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(digit_folder, ext)))

        # We limit it to max_samples
        self.image_paths = self.image_paths[:max_samples]
        self.target_label = target_label

        print(f"Found {len(self.image_paths)} images for digit {target_label}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # We load image only when requested !
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        image = image.to(device)
        label = torch.tensor(self.target_label, device=device)

        return image, label


def load_local_digit_subset_optimized(data_dir, target_label, max_samples, batch_size=1000, norm_factor=1.0):
    """
    Efficiently loads only the requested number of images from a specific digit folder.

    Parameters:
    - data_dir (str): Root directory containing folders 0-9.
    - target_label (int): The digit class to load (0-9).
    - max_samples (int): The number of samples to load from that class.
    - batch_size (int): Batch size for DataLoader.
    - norm_factor (float): Divide pixel values by this.

    Returns:
    - DataLoader of selected digit samples.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensures 1 channel
        transforms.Resize((28, 28)),  # Resizes to MNIST size
        transforms.ToTensor(),  # Converts to tensor [0,1]
        transforms.Lambda(lambda x: x.view(-1) / norm_factor)  # Flattens (784) and normalizes
    ])

    dataset = OptimizedDigitDataset(
        data_dir=data_dir,
        target_label=target_label,
        max_samples=max_samples,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":

    print("hello ,", time())
    start = time()

    loader = load_local_digit_subset_optimized(
        data_dir=Config.PROCESSED_TRAIN_PATH,
        target_label=5,
        max_samples=100,
        batch_size=100,
        norm_factor=1.0
    )

    print("Finished loading", time() - start)

    for batch in loader:
        images, labels = batch
        print("Batch shape:", images.shape, "Device:", images.device)
        break

    # Visualization
    for images, labels in loader:
        # images: shape [batch_size, 784]
        # Reshape images to 28x28 for display
        num_to_show = min(len(images), 10)
        images_to_show = images[:num_to_show].reshape(-1, 28, 28)
        labels_to_show = labels[:num_to_show]

        # Plot
        plt.figure(figsize=(2 * num_to_show, 2))
        for i in range(num_to_show):
            plt.subplot(1, num_to_show, i + 1)
            plt.imshow(images_to_show[i].cpu().numpy(), cmap='gray')
            plt.title(str(labels_to_show[i].item()))
            plt.axis('off')
        plt.suptitle(f"Samples of Digit Class {labels_to_show[0].item()}")
        plt.tight_layout()
        plt.show()
        break  # only shows one batch