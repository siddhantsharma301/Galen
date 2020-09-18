"""
VGG Network for Galen model
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Encoder network for Galen VGG style architecture
    """
    def __init__(self, latent_dim = 512):
        super(Encoder, self).__init__()
        self.latent = latent_dim
        self.padding1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 64, (7, 7))
        self.conv2 = nn.Conv2d(64, 128, (3, 3), stride = 2)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), stride = 2)
        self.conv4 = nn.Conv2d(256, 512, (3, 3), stride = 2)
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


class Decoder(nn.Module):
    """
    Decoder network for Galen
    """
    def __init__(self, latent_dim = 512):
        super(Decoder, self).__init__()
        self.latent = latent_dim
        self.fc1 = nn.Linear(self.latent, 64 * 64 * 64)
        self.conv_transpose = nn.ConvTranspose2d(64, 32, 3, stride = (2,2), padding_mode = 'zeros', output_padding = 1)
        self.conv = nn.Conv2d(32, 3, 3)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, 64, 64)
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.sig(x)


class VAE(nn.Module):
    """
    Galen model, combining encoder and decoder
    """
    def __init__(self, latent_dim=512, device='cpu', init_weights=True):
        super(VAE, self).__init__()
        self.latent = latent_dim
        self.device = device
        self.encoder = Encoder(self.latent).apply(self.weights_init if init_weights else None)
        self.decoder = Decoder(self.latent).apply(self.weights_init if init_weights else None)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        t_mean, t_logvar = self.encoder(x)
        std = t_logvar.mul(0.5).exp_()
        multi_norm = Variable(torch.FloatTensor(std.size()).normal_().to(self.device))
        z = multi_norm.mul(std).add_(t_mean)
        return self.decoder(z), t_mean, t_logvar

    def reconstruct(self, x):
        x, _, _ = self.forward(x)
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()


if __name__ == '__main__':
    # Test everything works
    from torchsummary import summary 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = VAE(device=device)
    summary(vae, input_size=(3, 128, 128))
