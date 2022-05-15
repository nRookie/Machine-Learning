import torch
import torch.nn as nn
import pandas
import random
import numpy
import matplotlib.pyplot as plt

from Discriminator import Discriminator
from Generator import Generator




def generate_real():
    real_data = torch.FloatTensor(
        [random.uniform(0.0, 0.2),
         random.uniform(0.8, 1.0),
         random.uniform(0.8, 1.0),
         random.uniform(0.8, 1.0),
         random.uniform(0.0, 0.2)])
    return real_data

def generate_random(size):
    random_data = torch.rand(size)
    return random_data

# create Discriminator and Generator
D = Discriminator()
G = Generator()

# define an image list to store how output evolves during training
image_list = []


# train Discriminator and Generator
for i in range(10000):
    
    # train discriminator on true
    D.train(generate_real(), torch.FloatTensor([1.0]))
    
    # train discriminator on false
    # use detach() so gradients in G are not calculated
    D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
    
    # train generator
    G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
    
    # add image to list every 1000
    if (i % 1000 == 0):
      image_list.append( G.forward(torch.FloatTensor([0.5])).detach().numpy() )

    pass


# plot images collected during training
plt.figure(figsize = (16,8))
plt.imshow(numpy.array(image_list).T, interpolation='none', cmap='Blues')
plt.savefig('output/legend.png')