import torch
import torch.nn as nn
from torch.utils.data import Dataset


import pandas, numpy, random
import matplotlib.pyplot as plt
from Dataset import FMnistDataset
from Discriminator import Discriminator


# load data
fmnist_dataset = FMnistDataset('fashion-mnist_train.csv')

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

D = Discriminator()
G = Generator()

epochs = 4 

for epoch in range(epochs):
    print("epoch = ", epoch + 1)
    # train Discriminator and Generator
    for label, image_data_tensor, target_tensor in fmnist_dataset:
        # train discriminator on true
        D.train(image_data_tensor, torch.FloatTensor([1.0]))

        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))

        pass


# plot several outputs from the trained generator
# plot a 3 column, 2 row array of generated images

f, axarr = plt.subplots(2,3, figsize=(16, 8))

for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100))
        img = output.detach().numpy().reshape(28,28)
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass
