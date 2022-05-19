import torchvision.datasets
mnist_dataset = torchvision.datasets.CelebA(root='.', download=True)

import h5py
import zipfile
import imageio
import os

hdf5_file = '/root/neural/HumanFace/celeba_dataset/celeba_aligned_small.h5py'

# how many of the 202,599 images to extract and package into HDF5
total_images = 20000


with h5py.File(hdf5_file, 'w') as hf:
    count = 0

    with zipfile.ZipFile('celeba/img_align_celeba.zip', 'r') as zf:
        for i in zf.namelist():
            if (i[-4:] == '.jpg'):
                # extract image
                ofile = zf.extract(i)
                img = imageio.imread(ofile)
                os.remove(ofile)
                
                # add image data to HDF5 file with new name
                hf.create_dataset('img_align_celeba/' + str(count) + '.jpg', data=img, compression="gzip", compression_opts=9)

                count = count + 1

                if (count% 1000 == 0):
                    print("images done ..", count)
                    pass
                # stop when total_images reached
                if (count == total_images):
                    break
                pass
            pass
        pass

