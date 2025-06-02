# Hopfield Networks - Developed by Alvin Kollçaku (2025)
# Licensed under the GNU General Public License v3.0

import os
import struct
import numpy as np
from PIL import Image

def load_images(path):
    with open(path, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def load_labels(path):
    with open(path, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def save_images(images, labels, dataset_type='train'):
    base_path = f'C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\processed\\{dataset_type}'
    os.makedirs(base_path, exist_ok=True)

    for i in range(10):
        os.makedirs(os.path.join(base_path, str(i)), exist_ok=True)

    for idx, (img, label) in enumerate(zip(images, labels)):
        img_path = os.path.join(base_path, str(label), f"{idx}.png")
        Image.fromarray(img).save(img_path)

train_images = load_images("C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\raw\\train-images.idx3-ubyte")
train_labels = load_labels("C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\raw\\train-labels.idx1-ubyte")
test_images = load_images("C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\raw\\test-images.idx3-ubyte")
test_labels = load_labels("C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\raw\\test-labels.idx1-ubyte")

# Reorganizing
save_images(train_images, train_labels, 'train')
save_images(test_images, test_labels, 'test')
