import numpy as np
from torchvision import datasets, transforms

mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)
test_images = mnist_test.data.numpy().astype(np.float32) / 255.0
train_images = mnist_train.data.numpy().astype(np.float32) / 255.0
test_labels = mnist_test.targets.numpy().astype(np.float32)
train_labels = mnist_train.targets.numpy().astype(np.float32)

train_images.tofile("data/mnist_train_images.mat")
test_images.tofile("data/mnist_test_images.mat")
train_labels.tofile("data/mnist_train_labels.mat")
test_labels.tofile("data/mnist_test_labels.mat")

print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
print(test_labels.shape)
