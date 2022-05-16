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
print("Discriminator:\n", D)

print("Training the discriminator")
for label, image_data_tensor, target_tensor in fmnist_dataset:
    # real data
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    # fake data
    D.train(generate_random_image(784), torch.FloatTensor([0.0]))

print("Checking the discriminator output")
print("The real input:\n")

for i in range(4):
    image_data_tensor = fmnist_dataset[random.randint(0,140)][1]
    print( D.forward( image_data_tensor).item())
    pass


print("Checking the discriminator output")
print("The real input:\n")

for i in range(4):
    image_data_tensor = fmnist_dataset[random.randint(0, 140)][1]
    print( D.forward( image_data_tensor).item())
    pass


print("The fake input:\n")

for i in range(4):
    print( D.forward( generate_random_image(784)).item())
    pass