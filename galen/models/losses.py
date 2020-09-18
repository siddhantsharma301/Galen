"""
Loss Functions for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import torch
import torch.nn as nn
import torchvision.models as models


# Layers in VGG-19 model
# In format: block<number>_<layer_type><layer_number>
# Example: b1_c1 --> block 1, conv 1
layer_names = ['b1_c1', 'b1_r1', 'b1_c2', 'b1_r2', 'b1_p',
               'b2_c1', 'b2_r1', 'b2_c2', 'b2_r2', 'b2_o',
               'b3_c1', 'b3_r1', 'b3_c2', 'b3_r2', 'b3_c3', 'b3_r3', 'b3_c4', 'b3_r4', 'b3_p',
               'b4_c1', 'b4_r1', 'b4_d2', 'b4_r2', 'b4_c3', 'b4_r3', 'b4_c4', 'b4_r4', 'b4_p',
               'b5_c1', 'b5_r1', 'b5_c2', 'b5_r2', 'b5_c3', 'b5_r3', 'b5_c4', 'b5_r4', 'b5_p']
# Specific content layers from VGG-19
vae123_layers = ['b1_r1', 'b2_r1', 'b3_r1']
vae234_layers = ['b2_r1', 'b3_r1', 'b4_r1']
vae345_layers = ['b3_r1', 'b4_r1', 'b5_r1']
vae1234_layers = ['b1_r1', 'b2_r1', 'b3_r1', 'b4_r1']


class VGGLossModel(nn.Module):
    """
    VGG model for perceptual loss
    """
    def __init__(self, content_layers):
        super(VGGLossModel, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential()

        for i, module in enumerate(features):
            name = layer_names[i]
            self.layers.add_module(name, module)

        # Turn off training for perceptual model
        for param in features.parameters():
            param.requires_grad = False

        # Content layers
        if content_layers == "vae-123":
            self.content_layers = vae123_layers
        elif content_layers == "vae-234":
            self.content_layers = vae234_layers
        elif content_layers == "vae-345":
            self.content_layers = vae345_layers
        elif content_layers == 'vae-1234':
            self.content_layers = vae1234_layers
        

    def forward(self, inputs):
        batch_size = inputs.size(0)
        all_outputs = []
        output = inputs
        for name, module in self.layers.named_children():
            output = module(output)
            if name in self.content_layers:
                all_outputs.append(output.view(batch_size, -1))
        return all_outputs


class VGGPerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss for Galen training
    """
    def __init__(self, content_layers, reduction='sum', device='cpu'):
        super(VGGPerceptualLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.vgg = VGGLossModel(content_layers).to(device)
    
    def forward(self, orig, recon):
        orig_features = self.vgg(orig)
        recon_features = self.vgg(recon)
        return self.loss(recon_features, orig_features)

    def loss(self, recon_features, orig_features):
        loss = 0
        for recon, orig in zip(recon_features, orig_features):
            loss += self.criterion(recon, orig)
        return loss


class KLDLoss(nn.Module):
    """
    KLD Loss module for Galen training
    """
    def __init__(self, reduction='sum'):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mean, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
        if self.reduction == 'mean':
            kld_loss = torch.mean(kld_loss)
        elif self.reduction == 'sum':
            kld_loss = torch.sum(kld_loss)
        return kld_loss


if __name__ == "__main__":
    # Test everything works
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    perceptual_loss_criterion = VGGPerceptualLoss('vae-1234', device=device) 
    kld_loss_criterion = KLDLoss()

    print("Everything's good so far!")
