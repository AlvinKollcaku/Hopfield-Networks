# Hopfield Networks - Developed by Alvin Kollçaku (2025)
# Licensed under the GNU General Public License v3.0

import matplotlib.pyplot as plt
import struct
import numpy as np

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images

images = load_mnist_images("C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\raw\\train-images.idx3-ubyte")
print(images.shape)  # Output: (60000, 28, 28)

# Plotting the first 10 images
plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.suptitle("First 10 MNIST Digits", fontsize=16)
plt.show()


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

labels = load_mnist_labels("C:\\Users\\alvin\\OneDrive\\Desktop\\Bachelor_Thesis\\MNIST\\raw\\train-labels.idx1-ubyte")
print("First 10 labels:", labels[:10])
