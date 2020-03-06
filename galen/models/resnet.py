#!/usr/bin/env python3

from functools import partial
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['sig', nn.Sigmoid()],
        ['none', nn.Identity()]])[activation]


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dAuto, self).__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)  


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels))


class ResidualBase(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(ResidualBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResidualDownBlock(ResidualBase):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super(ResidualDownBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
              
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class BasicDownBlock(ResidualDownBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(BasicDownBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False))


class ResidualDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=BasicDownBlock, n=1, *args, **kwargs):
        super(ResidualDownLayer, self).__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)])

    def forward(self, x):
        x = self.blocks(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], 
                 activation='relu', block=BasicDownBlock, latent_dim=512, *args, **kwargs):
        super(Encoder, self).__init__()
        self.blocks_sizes = blocks_sizes
        self.latent_dim = latent_dim
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResidualDownLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResidualDownLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]])

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.mu = nn.Linear(self.blocks[-1].blocks[-1].expanded_channels, self.latent_dim)
        self.logvar = nn.Linear(self.blocks[-1].blocks[-1].expanded_channels, self.latent_dim)
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar



class Deconv2dAuto(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(Deconv2dAuto, self).__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


convtranspose3x3 = partial(Deconv2dAuto, kernel_size=3, bias=False)  


def convtranspose_bn(in_channels, out_channels, convtranspose, *args, **kwargs):
    return nn.Sequential(convtranspose(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels))


class ResidualUpBlock(ResidualBase):
    def __init__(self, in_channels, out_channels, expansion=1, upsampling=2, conv=convtranspose3x3, *args, **kwargs):
        super(ResidualUpBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.upsampling = upsampling
        self.conv = conv
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.upsampling, bias=False, padding_mode='zeros', output_padding=1),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class BasicUpBlock(ResidualUpBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, use_padding=False, *args, **kwargs):
        super(BasicUpBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            convtranspose_bn(self.in_channels, self.out_channels, convtranspose=self.conv, bias=False, stride=self.upsampling),
            activation_func(self.activation),
            convtranspose_bn(self.out_channels, self.expanded_channels, convtranspose=self.conv, bias=False),
            nn.ZeroPad2d((1, 0, 1, 0)) if use_padding else nn.ZeroPad2d((0, 0, 0, 0)))


class ResidualUpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=BasicUpBlock, n=1, use_padding=False, *args, **kwargs):
        super(ResidualUpLayer, self).__init__()
        upsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, use_padding=use_padding, *args, **kwargs, upsampling=upsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, upsampling=1, *args, **kwargs) for _ in range(n - 1)])

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, blocks_sizes=[512, 256, 128, 64, 3], depths=[2,2,2,2,2], 
                 activation='relu', block=BasicUpBlock, *args, **kwargs):
        super(Decoder, self).__init__()

        self.blocks_sizes = blocks_sizes
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 512 * 7 * 7)
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResidualUpLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, 
                        block=block, use_padding=False, *args, **kwargs),
            *[ResidualUpLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, use_padding=True,*args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]])

        self.gate = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(3),
            activation_func('sig'))
        
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        for block in self.blocks:
            x = block(x)
        x = self.gate(x)
        return x



class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, down_block=BasicDownBlock, 
                 up_block=BasicUpBlock, encoder_depths=[2,2,2,2], decoder_depths=[2,2,2,2,2],
                 device='cpu', init_weights=True):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.device = device

        self.encoder = Encoder(in_channels=in_channels, depths=encoder_depths).to(self.device).apply(self.weights_init if init_weights else None)
        self.decoder = Decoder(latent_dim=latent_dim, depths=decoder_depths).to(self.device).apply(self.weights_init if init_weights else None)

        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

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

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        multi_norm = Variable(torch.FloatTensor(std.size()).normal_().to(self.device))
        z = multi_norm.mul(std).add_(mu)
        return z 

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        return self.decoder(z), mean, logvar

    def reconstruct(self, x):
        x, _, _ = self.forward(x)
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).children())[:-1])

        for name, params in self.model.named_parameters():
            params.requires_grad = False

        self.fc = nn.Linear(512 * 7 * 7, 2)
        self.sig = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return self.sig(self.fc(x))
    
    def predict(self,x):
        outputs = self.forward(x)
        return torch.max(outputs, 1)
    

if __name__ == "__main__":
    # Test everything works
    from torchsummary import summary

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = VAE(device=device)
    summary(vae, input_size=(3, 224, 224))

    print('\n\n')

    disc = Discriminator().to(device)
    summary(disc, input_size=(3, 224, 224))
