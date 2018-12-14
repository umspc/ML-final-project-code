# author: Wenbo Yu
# Note: this code is based on a PyTorch platform open source in github @geniki.

from NN import NN
from NP import NP
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import random
import os
from skimage.io import imread, imsave
import numpy as np
import cv2

IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 128
EPOCH = 300
num_pix = IMG_WIDTH * IMG_HEIGHT

def creatPath(directory):
    files = os.listdir(directory)
    fileList = []
    for fileName in files:
        filePath = os.path.join(directory, fileName)
        fileList.append(filePath)
    return fileList

def loadImg(path):
    imgList = []
    for imgName in path:
        img = imread(imgName)
        # img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        # img = color.rgb2gray(img)
        imgList.append(img)
        # imsave(imgName[24:], img)
    imgList = np.array(imgList, dtype = 'float32')
    return imgList

torch.manual_seed(1)
random.seed(1)
device = torch.device("cuda")

# create image path
colorList_train = creatPath('../test')
colorList_test = creatPath('../test')

#load image
print 'Loading images ...'
colorImg_train = loadImg(colorList_train)
colorImg_test = loadImg(colorList_test)
colorImg_train = colorImg_train / 255.
colorImg_test = colorImg_test / 255.

num_train = colorImg_train.shape[0]
num_test = colorImg_test.shape[0]
colorImg_train = torch.from_numpy(colorImg_train)
colorImg_test = torch.from_numpy(colorImg_test).cuda()
print 'Finish loading images'

def get_context_idx(N):
    idx = random.sample(range(0, num_pix), N)
    idx = torch.tensor(idx, device=device)
    return idx

def generate_grid(h, w):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid

def idx_to_y(idx, data):
    y = torch.index_select(data, dim=1, index=idx)
    return y

def idx_to_x(idx, batch_size):
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x


def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div


def np_loss(y_hat, y, z_all, z_context):
    loss1 = F.binary_cross_entropy(y_hat, y, reduction="sum")
    loss2 = kl_div_gaussians(z_all[0], z_all[1], z_context[0], z_context[1])
    return loss1 + loss2

x_grid = generate_grid(IMG_HEIGHT, IMG_WIDTH)
model = NN(x_grid).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# training
model.train()
train_loss = 0
for i in range(EPOCH):
    for batch_idx in range(int(num_train / BATCH_SIZE)):
        batch_order = random.sample(range(1, num_train), BATCH_SIZE)
        y_all = colorImg_train[batch_order,:]
        y_all = y_all.cuda()
        y_all= torch.reshape(y_all, (BATCH_SIZE, -1, 1))
        N = random.randint(1, num_pix)
        context_idx = get_context_idx(N)
        x_context = idx_to_x(context_idx, BATCH_SIZE)
        y_context = idx_to_y(context_idx, y_all)
        x_all = x_grid.expand(BATCH_SIZE, -1, -1)

        optimizer.zero_grad()
        y_hat, z_all, z_context = model(x_context, y_context, x_all, y_all)

        loss = np_loss(y_hat, y_all, z_all, z_context)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch: ', i)
            print('Batch: ', batch_idx)
            print('loss: ', loss.item())
    if (i+1) > 100 and (i+1) % 20 == 0:
        weight_name = 'weight/epoch_%d.pkl' % (i+1)
        torch.save(model.state_dict(), weight_name)


# test
model.eval()
test_loss = 0
with torch.no_grad():

    for batch_idx in range(num_test):

        batch_order = random.sample(range(1, num_train), BATCH_SIZE)
        y_all = colorImg_train[batch_order, :]
        y_all = y_all.cuda()
        y_all = torch.reshape(y_all, (BATCH_SIZE, -1, 1))

        N = 300
        context_idx = get_context_idx(N)
        x_context = idx_to_x(context_idx, BATCH_SIZE)
        y_context = idx_to_y(context_idx, y_all)

        y_hat, z_all, z_context = model(x_context, y_context)
        test_loss += np_loss(y_hat, y_all, z_all, z_context).item()

        if batch_idx == 0:
            plot_Ns = [10, 200, 800, 1024]
            num_examples = min(BATCH_SIZE, 16)
            for N in plot_Ns:
                recons = []
                context_idx = get_context_idx(N)
                x_context = idx_to_x(context_idx, BATCH_SIZE)
                y_context = idx_to_y(context_idx, y_all)
                for d in range(5):
                    y_hat, _, _ = model(x_context, y_context)
                    recons.append(y_hat[:num_examples])
                recons = torch.cat(recons).view(-1, 1, 32, 32).expand(-1, 3, -1, -1)
                background = torch.tensor([0., 0., 1.], device=device)
                background = background.view(1, -1, 1).expand(num_examples, 3, 1024).contiguous()
                context_pixels = y_all[:num_examples].view(num_examples, 1, -1)[:, :, context_idx]
                context_pixels = context_pixels.expand(num_examples, 3, -1)
                background[:, :, context_idx] = context_pixels
                comparison = torch.cat([background.view(-1, 3, 32, 32), recons])

