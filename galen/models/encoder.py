"""
Encoder Network for Galen model
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network for Galen
    """
    def __init__(self, latent_dim = 512):
        super(Encoder, self).__init__()
        self.latent = latent_dim
        self.padding1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 64, (7, 7), padding_mode = 'valid')
        self.conv2 = nn.Conv2d(64, 128, (3, 3), stride = 2, padding_mode = 'valid')
        self.conv3 = nn.Conv2d(128, 256, (3, 3), stride = 2, padding_mode = 'valid')
        self.conv4 = nn.Conv2d(256, 512, (3, 3), stride = 2, padding_mode = 'valid')
        self.t_mean = nn.Linear(512 * 15 * 15, self.latent)
        self.t_logvar = nn.Linear(512 * 15 * 15, self.latent)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.padding1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = x.view(-1, 512 * 15 * 15)
        t_mean = self.t_mean(x)
        t_logvar = self.t_logvar(x)
        return t_mean, t_logvar
