"""
Decoder Network for Galen model
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder network for Galen
    """
    def __init__(self, latent_dim = 512):
        super(Decoder, self).__init__()
        self.latent = latent_dim
        self.fc1 = nn.Linear(self.latent, 64 * 64 * 64)
        self.conv_transpose = nn.ConvTranspose2d(64, 32, 3, stride = (2,2), padding_mode = 'zeros', output_padding = 1)
        self.conv = nn.Conv2d(32, 3, 3, padding_mode = 'same')
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, 64, 64)
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.sig(x)
