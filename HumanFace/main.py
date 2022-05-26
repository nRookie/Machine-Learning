import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt
import numpy

from Discriminator import Discriminator
from Generator import Generator

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data



D = Discriminator()

hdf5_file = '/root/neural/HumanFace/celeba_dataset/celeba_aligned_small.h5py'

if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D.to(device)


from celebadataset import  CelebADataset

celeba_dataset = CelebADataset(hdf5_file)

for image_data_tensor in celeba_dataset:
    # real data
    D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))
    # fake data
    D.train(generate_random_image((218, 178, 3)), torch.cuda.FloatTensor([0.0]))
    pass


D.plot_progress()

# plt.show()


G = Generator()

G.to(device)



output = G.forward(generate_random_seed(100))
img = output.detach().cpu().numpy()
plt.imshow(img, interpolation='none', cmap='Blues')

'''
The code is very familiar now. We create a new generator object and move it to the GPU. We feed the generator a random seed and generate an output. Before we can display that output as an image, we need to detach() it from PyTorch’s computation graph, move it from the GPU to the CPU, and then convert it to a NumPy array.
'''