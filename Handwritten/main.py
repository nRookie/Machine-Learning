import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas, numpy, random
import matplotlib.pyplot as plt


class MnistDataset(nn.Module):
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # image target (label)
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0

        # image data, normalised from 0-255 to 0 - 1
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

        # return label, image data tensor and 0-255 to 0-1
        return label, image_values, target
    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))

        plt.imshow(img, interpolation='none', cmap='Blues')
        pass
    pass

class Discriminator(nn.Module):
	def __init__(self):
		# initialise parent pytorch class
		super().__init__()

		# define neural network layers
		self.model = nn.Sequential(
			nn.Linear(784, 200),
			nn.Sigmoid(),
			nn.Linear(200, 1),
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



class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 784),
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



D = Discriminator()


mnist_dataset = MnistDataset('../mnist_train.csv')

mnist_dataset.plot_image(17)


def generate_random(size):
    random_data = torch.rand(size)
    return random_data

# G = Generator()
# output = G.forward(generate_random(1))
# img = output.detach().numpy().reshape(28,28)
# plt.imshow(img,interpolation='none', cmap='Blues')
# plt.show()
# # for label, image_data_tensor, target_tensor in mnist_dataset:
# #     # real data
# #     D.train(image_data_tensor, torch.FloatTensor([1.0]))
# #     # fake data
# #     D.train(generate_random(784), torch.FloatTensor([0.0]))
# #     pass


# # D.plot_progress()


# # for i in range(4):
# #   image_data_tensor = mnist_dataset[random.randint(0,60000)][1]
# #   print( D.forward( image_data_tensor ).item() )
# #   pass

# # for i in range(4):
# #   print( D.forward( generate_random(784) ).item() )
# #   pass