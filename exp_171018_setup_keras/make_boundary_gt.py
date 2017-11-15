import numpy as np
import h5py
from matplotlib import pyplot as plt

data_filepath = '/media/julian/Daten/datasets/cremi_2016/cremi.splA.train.raw_neurons.crop.h5'
data_targetpath = '/media/julian/Daten/datasets/cremi_2016/exp_171018_setup_keras/cremi.splA.train.boundaries.h5'

with h5py.File(data_filepath, 'r') as f:
    raw = np.array(f['raw'])
    labels = np.array(f['neuron_ids'])

print(raw.shape)
print(labels.shape)

from skimage.segmentation import find_boundaries

boundaries = np.array([find_boundaries(labels[z]) for z in range(labels.shape[0])])

plt.imshow(boundaries[20, :, :])
plt.show()

with h5py.File(data_targetpath, 'w') as f:
    f['data'] = boundaries

