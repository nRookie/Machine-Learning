import torch
import torch.nn as nn

def generate_random(size):
	random_data = torch.rand(size)
	return random_data


print("Random noise pattern:", generate_random(4))
