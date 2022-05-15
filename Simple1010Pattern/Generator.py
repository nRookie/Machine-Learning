import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt
import numpy

class Discriminator(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        # create loss function
        self.loss_function = nn.MSELoss()

        # create optimiser, using stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

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



def generate_real():
    real_data = torch.FloatTensor([1, 0, 1, 0])
    return real_data

def generate_random(size):
    random_data = torch.rand(size)
    return random_data


class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []
        
        pass
    
    
    def forward(self, inputs):        
        # simply run model
        return self.model(inputs)

    def train(self, D, inputs, targets):
        # calculate the output of the network
        g_output = self.forward(inputs)
            
        # pass onto Discriminator
        d_output = D.forward(g_output)
            
        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
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

# creating discriminator object
D = Discriminator()

# creating generator object
G = Generator()
print("Output of untrained generator:", G.forward(torch.FloatTensor([0.5])))

image_list = []

for i in range(10000):
    if(i % 1000 == 0):
        image_list.append( G.forward(torch.FloatTensor([0.5])).detach().numpy() )
    # train discriminator on false
    D.train(generate_real(), torch.FloatTensor([1.0]))
    # train discriminator on false
    # use detach() so gradients in G are not calculated
    D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))

    # train generator
    G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
    pass


D.plot_progress()
plt.xlabel('Discriminator loss chart')
plt.savefig('output/Discriminator.png')

G.plot_progress()
plt.xlabel('Generator loss chart')
plt.savefig('output/Generator.png')


print("Output of trained generator:", G.forward(torch.FloatTensor([0.5])))

plt.figure(figsize = (16,8))
plt.imshow(numpy.array(image_list).T, interpolation='none', cmap='Blues')
plt.show()