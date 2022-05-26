import torch
import torch.nn as nn

import h5py

import numpy
import matplotlib.pyplot as plt


from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']
        pass
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        if (index >= len(self.dataset)):
            raise IndexError()
        img = numpy.array(self.dataset[str(index) +'.jpg'])
        return torch.cuda.FloatTensor(img)  / 255.0

    def plot_image(self, index):
        plt.imshow(numpy.array(self.dataset[str(index) + '.jpg']), interpolation='nearest')

    pass


# hdf5_file = '/root/neural/HumanFace/celeba_dataset/celeba_aligned_small.h5py'

# celeba_dataset = CelebADataset(hdf5_file)
# celeba_dataset.plot_image(33)
# plt.show()