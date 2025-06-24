# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie (modifié par ChatGPT pour signal complexe)
@File: mutual_info.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mine(nn.Module):
    def __init__(self, in_dim=4, hidden_size=64):
        super(Mine, self).__init__()
        self.dense1 = linear(in_dim, hidden_size)
        self.dense2 = linear(hidden_size, hidden_size)
        self.dense3 = linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        return self.dense3(x)

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin.weight = torch.nn.Parameter(torch.normal(0.0, 0.02, size=lin.weight.shape))
    if bias:
        lin.bias.data.zero_()
    return lin

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et

def learn_mine(batch, mine_net, ma_et, ma_rate=0.01):
    joint, marginal = batch
    joint = joint.float()
    marginal = marginal.float()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
    loss = -(torch.mean(t) - (1 / torch.mean(ma_et)) * torch.mean(et))
    return loss, ma_et, mi_lb

def sample_batch(x_in, y_out):
    # Signal complexe : on transforme x et y en (Re, Im)
    x_real = x_in.real.reshape(-1, 1)
    x_imag = x_in.imag.reshape(-1, 1)
    x = torch.cat([x_real, x_imag], dim=1)

    y_real = y_out.real.reshape(-1, 1)
    y_imag = y_out.imag.reshape(-1, 1)
    y = torch.cat([y_real, y_imag], dim=1)

    # On coupe en deux pour faire les paires
    x1, x2 = torch.chunk(x, 2, dim=0)
    y1, y2 = torch.chunk(y, 2, dim=0)

    joint = torch.cat([x1, y1], dim=1)     # vrais couples
    marginal = torch.cat([x1, y2], dim=1)  # y mélangé → indépendants

    return joint, marginal
