
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


class ClassDataset(Dataset):

    def __init__(self, data, noisy):
        self.data = data
        self.noisy = noisy  
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.noisy[idx]

        return data, target


if __name__ == "__main__":
    original = np.load('./NoisyMNIST/train_original.npy').astype(np.float32)
    noisy = np.load('./NoisyMNIST/train_noisy.npy').astype(np.float32)

    dataset = ClassDataset(original, noisy)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=False)
    #data_bar = tqdm(dataloader)
    for data, target in dataloader:
        
        print(data)
        print('======')
        
        break

    for data, target in dataloader:
        
        print(data)
        print('======')
        
        break
    print('\n')
    data, target = next(iter(dataloader))
    print(data)
   