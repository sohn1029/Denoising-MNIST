import torch

import torchvision
import torchvision.transforms as transforms
from UNet_S import ContextUnet
from ddpm import DDPM

import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as img
from score_utils import *
from data import ClassDataset
from torch.utils.data import Dataset, DataLoader

# CUDA_VISIBLE_DEVICES=2 python /home/geonho/unlearning_mlp/train.py

#hyper parameter
epochs = 50
lr = 0.0003

batch_size = 128
n_T = 500 # 500
device = "cuda:0"

test_batch_size = 10

n_classes = 10
n_feat = 256 # 128 ok, 256 better (but slower)

save_model = True
model_save_dir = '/media/data1/geonho/ddpm_mnist/'
img_save_dir = './result_img/'
ws_test = [0.0, 0.5, 2.0] # strength of generative guidance'

data_shape = (1, 28, 28)
transform = transforms.Compose([
    transforms.ToTensor(),
])

#trainset = torchvision.datasets.MNIST('/media/data1/data/MNIST', train = True, transform=transform, download=True)
#trainset = torchvision.datasets.CIFAR10('/media/data1/data', train = True, transform=transform, download=True)

train_original = np.load('./NoisyMNIST/train_original.npy').astype(np.float32)
train_noisy = np.load('./NoisyMNIST/train_noisy.npy').astype(np.float32)

trainset = ClassDataset(train_original, train_noisy)


test_original = np.load('./NoisyMNIST/test_original.npy').astype(np.float32)
test_noisy = np.load('./NoisyMNIST/test_noisy.npy').astype(np.float32)

testset = ClassDataset(test_original, test_noisy)

print(type(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)




#testset = torchvision.datasets.MNIST('/media/data1/data/MNIST', train = False, transform=transform, download = True)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)



#model
cuda = torch.device('cuda:0')
model = DDPM(nn_model=ContextUnet(in_channels=data_shape[0], n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).cuda()

#loss function
criterion = nn.CrossEntropyLoss().cuda()

#optimizer
optimizer = optim.Adam(model.parameters(), lr= lr)

for epoch in range(epochs):
    #train
    model.train()
    train_bar = tqdm(trainloader)
    optimizer.param_groups[0]['lr'] = lr*(1-epoch/epochs)
    loss_ema = None
    for data, target in train_bar:

        #gradient init
        optimizer.zero_grad()

        #gpu load
        data = data.cuda()
        target = target.cuda()

        #predict
        loss = model(data, target)
        
        #gradient stack
        loss.backward()
        
        if loss_ema is None:
                loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
    
        train_bar.set_description(f"loss: {loss_ema:.4f}")

        #gradient apply
        optimizer.step()

        #**********
        #break
        #**********

    #validation
    model.eval()
    with torch.no_grad():
        

        test_data, test_target = next(iter(testloader))
        test_data = test_data.to(device)
        test_target = test_target.to(device)

        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = model.sample(test_target, data_shape, device, guide_w=w)
            
            # append some real images at bottom, order by class also
            # x_real = torch.Tensor(x_gen.shape).to(device)
            x_real = test_target
            # for k in range(n_classes):
                
            #     for j in range(int(n_sample/test_batch_size)):
            #         try: 
            #             idx = torch.squeeze((target == k).nonzero())[j]
            #         except:
            #             idx = 0
                    
            #         x_real[k+(j*n_classes)] = data[idx]

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all, nrow=10)
            save_image(grid, img_save_dir + f"image_ep{epoch}_w{w}.png")
            print('saved image at ' + img_save_dir + f"image_ep{epoch}_w{w}.png")


        
            if epoch%5==0 or epoch == int(epochs-1):
                # create gif of images evolving over time, based on x_gen_store
                fig, axs = plt.subplots(nrows=int(1), ncols=test_batch_size,sharex=True,sharey=True,figsize=(8,3))
                def animate_diff(i, x_gen_store):
                    print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    plots = []
                    
                    for col in range(test_batch_size):
                        axs[col].clear()
                        axs[col].set_xticks([])
                        axs[col].set_yticks([])
                        # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                        
                        plots.append(axs[col].imshow(-x_gen_store[i,col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                    return plots
                print(x_gen_store.shape)
                ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                ani.save(img_save_dir + f"gif_ep{epoch}_w{w}.gif", dpi=100, writer=PillowWriter(fps=10))
                print('saved image at ' + img_save_dir + f"gif_ep{epoch}_w{w}.gif")
    
    
    # optionally save model
    if save_model and epoch == int(epochs-1):
        torch.save(model.state_dict(), model_save_dir + f"model_denoise_{epoch}.pth")
        print('saved model at ' + model_save_dir + f"model_denoise_{epoch}.pth")

     