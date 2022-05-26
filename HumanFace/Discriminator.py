import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt
import numpy


from celebadataset import  CelebADataset
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


class Discriminator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            View(218*178*3),

            nn.Linear(3 * 218 * 178, 100),
            nn.LeakyReLU(),

            nn.LayerNorm(100),

            nn.Linear(100, 1),
            nn.Sigmoid()
        )



        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, using stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []
        pass

    def forward(self, inputs):
        # simple run model
        return self.model(inputs)

    def train(self, inputs, targets):
        # calculate the output of the network 
        outputs = self.forward(inputs)
        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error ever 10 epochs

        self.counter += 1
        if (self.counter % 10 == 0 ):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass
        
        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass




def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


D = Discriminator()

hdf5_file = '/root/neural/HumanFace/celeba_dataset/celeba_aligned_small.h5py'

celeba_dataset = CelebADataset(hdf5_file)

for image_Data_tensor in celeba_dataset:
    # real data
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    # fake data
    D.train(generate_random_image((218, 178, 3)), torch.FloatTensor([0.0]))
    pass

D.plot_progress()

# plt.show()
