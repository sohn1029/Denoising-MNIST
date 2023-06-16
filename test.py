import torch
from ddpm import DDPM
from UNet_S import ContextUnet
import os
from torchvision.utils import save_image, make_grid

epochs = 20
lr = 0.0004

batch_size = 256
n_T = 500 # 500
device = "cuda:0"

n_classes = 10
n_feat = 256 # 128 ok, 256 better (but slower)

save_model = True
model_save_dir = '/media/data1/geonho/ddpm_mnist/model_cifar10_29.pth'
testroot = './test_result/'
ws_test = [0.0, 0.5, 2.0] # strength of generative guidance'
data_shape = (3, 32, 32)


print("Loading Model...")
cuda = torch.device('cuda:0')
model = DDPM(nn_model=ContextUnet(in_channels=data_shape[0], n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).cuda()
model.load_state_dict(torch.load(model_save_dir))
model.eval()

w = float(input("give w >> "))

testpath = testroot + str(w)
if not os.path.exists(testpath):
    os.mkdir(testpath)


with torch.no_grad():
    x_gen, x_gen_store = model.sample(40, data_shape, device, guide_w=w)
    
    x_gen_store = torch.Tensor(x_gen_store)
    for i in range(x_gen_store.shape[0]):
        grid = make_grid(x_gen_store[i], nrow=5)
        save_image(grid, testpath + '/' + f"test{i}.png")
