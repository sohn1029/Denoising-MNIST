import torch

import torchvision
import torchvision.transforms as transforms

from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img
from score_utils import *
from tqdm import tqdm
import sys

from scipy import signal
import random

np.set_printoptions(threshold=sys.maxsize)


device = "cuda:0"

data_shape = (3, 32, 32)

transform = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.MNIST('/media/data1/data/MNIST', train = True, transform = transform, download=True)



trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=4)

train_bar = tqdm(trainloader)



def gkern(kernlen=21, std_1=3, std_2=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d_1 = signal.gaussian(kernlen, std=std_1).reshape(kernlen, 1)
    gkern1d_2 = signal.gaussian(kernlen, std=std_2).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d_1, gkern1d_2)
    return gkern2d


def get_noise(c, img_size):
    ker_size = random.randint(3, img_size//2)
    std1 = random.randint(1, 5)
    std2 = random.randint(1, 5)
    noise_strength = random.randint(70, 95)/100
    noise = gkern(ker_size, std1, std2) * noise_strength
    noise = np.expand_dims(noise, axis = 0)
    noise = np.repeat(noise, c, axis = 0)
    
    x = random.randint(0, img_size-ker_size)
    y = random.randint(0, img_size-ker_size)

    return {'x': x,
            'y': y,
            'noise': noise}

def get_noisy_data(data, noise_list):
    for noise in noise_list:
        x, y = noise['x'], noise['y']
        ker_size = noise['noise'].shape[1]
        
        data[:,:, x:x+ker_size, y:y+ker_size] *= noise['noise']
    return data

for data, target in train_bar:
    data = data.numpy()
    b, c, w, h = data.shape
    
    noise_num = random.randint(3, 6)
    noise_arr = [get_noise(c, w) for _ in range(noise_num)]
    print(data.shape)
    noisy_data = get_noisy_data(data, noise_arr)[0]
    noisy_data = noisy_data.transpose((1, 2, 0))
    plt.imshow(noisy_data[0])
    plt.show()
    break
