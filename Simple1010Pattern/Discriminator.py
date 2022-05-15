import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt

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



D = Discriminator()
print("Training the discriminator")
for i in range(10000):
	# real data
	D.train(generate_real(), torch.FloatTensor([1.0]))
	# fake data
	D.train(generate_random(4), torch.FloatTensor([0.0]))
	pass

D.plot_progress()
plt.savefig('output/legend.png')

print("Real data source:", D.forward(generate_real()).item())

print("Random noise:", D.forward(generate_random(4)).item())
