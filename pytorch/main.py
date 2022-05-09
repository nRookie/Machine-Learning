import matplotlib.pyplot as plt
from Dataset import FMnistDataset
from NeuralNetwork import Classifier

fmnist_dataset = FMnistDataset('fashion-mnist_train.csv')

C = Classifier()
print("Classifer configuration:\n", C)
