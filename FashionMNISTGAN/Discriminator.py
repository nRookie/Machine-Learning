import torch
import torch.nn as nn
from torch.utils.data import Dataset

# discriminator class


class Discriminator(nn.Module):
    def __init__(self):
        # initialize parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LeakyNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        # create loss function
        self.loss_function = nn.BCELoss()
        # create optimiser, adam
        self.optimiser = torch.optim.Adam(self.parameters(), lr = 0.0001)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []
        pass
    def forward(self, inputs):
        # simpy run model
        return self.model(inputs)

    def train(self, inputs, targets):
        # calculate the output of the network
        output = self.forward(inputs)
        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if (self.counter % 10 == 0):
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

        

