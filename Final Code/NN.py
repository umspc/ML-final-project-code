# author: Shenyu Mou
# Note: this code is based on a PyTorch platform open source in github @geniki.

import torch
import torch.utils.data
from torch import nn


class NN(nn.Module):
    def __init__(self, x_grid):
        super(NN, self).__init__()
        self.r = 128
        self.z = 128
        self.x_grid = x_grid

        self.h = nn.Sequential(
            nn.Linear(3, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, self.r),
            nn.ReLU()
        )

        self.g = nn.Sequential(
            nn.Linear(self.z + 2, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

        self.r_to_z_mean = nn.Linear(self.r, self.z)
        self.r_to_z_logvar = nn.Linear(self.r, self.z)

    def stacastic(self, x, y):
        x_y = torch.cat([x, y], dim=2)
        r_i = self.h(x_y)
        r = torch.mean(r_i, dim=1)
        return r

    def forward(self, x_context, y_context, x_all=None, y_all=None):
        z_context = self.stacastic(x_context, y_context)
        if self.training:
            z_all = self.stacastic(x_all, y_all)
        else:
            z_all = z_context

        z_sample = z_all.unsqueeze(1).expand(-1, 1024, -1)
        x_target = self.x_grid.expand(y_context.shape[0], -1, -1)
        z_x = torch.cat([z_sample, x_target], dim=2)
        y_hat = self.g(z_x)

        return y_hat, z_all, z_context