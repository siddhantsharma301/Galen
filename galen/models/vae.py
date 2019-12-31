"""
VAE Network for Galen model
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import torch
import torch.nn as nn
from torch.autograd import Variable
from galen.models import encoder, decoder


class VAE(nn.Module):
    """
    Galen model, combining encoder and decoder
    """
    def __init__(self, latent_dim = 384, init_weights = True):
        super(VAE, self).__init__()
        self.latent = latent_dim
        self.encoder = encoder.Encoder(self.latent).apply(self.weights_init if init_weights else None)
        self.decoder = decoder.Decoder(self.latent).apply(self.weights_init if init_weights else None)

    def forward(self, x):
        t_mean, t_logvar = self.encoder(x)
        std = t_logvar.mul(0.5).exp_()
        multi_norm = Variable(torch.FloatTensor(std.size()).normal_().cpu())
        z = multi_norm.mul(std).add_(t_mean)
        return self.decoder(z), t_mean, t_logvar

    def encode(self, x):
        """
        Encode and sample latent space
        """
        t_mean, t_logvar = self.encoder(x)
        std = t_logvar.mul(0.5).exp_()
        multi_norm = Variable(torch.FloatTensor(std.size()).normal_().cpu())
        z = multi_norm.mul(std).add_(t_mean)
        return z

    def decode(self, x):
        return self.decoder(x)

    def weights_init(self, m):
        """
        A basic weights initializer for encoder
        and decoder networks
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


if __name__ == "__main__":
    vae = VAE()
    print("Done!")
