import torch
import random

def generate_real():
	real_data = torch.FloatTensor(
	[random.uniform(0.8, 1.0),
	random.uniform(0.0, 0.2),
	random.uniform(0.8, 1.0),
	random.uniform(0.0, 0.2)])
	return real_data
print("Real data:", generate_real())
