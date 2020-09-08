"""
Model Utilities for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import os
import glob
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model, weights=None, dir=None):
    """
    Load a trained model from indicated weights or dir

    Args:
    * weights
    	OPTIONAL
    	Path to weights (str)
    	Default: None
    * dir
    	OPTIONAL
    	Path to dir to find weights (str)
    	Default: './training/checkpoints/'

    Returns:
    Model with loaded weights (`torch.nn.module`)
    """
    if model == 'resnet':
        from galen.models import resnet
        model = resnet.VAE(device=device)
    elif model == 'vgg':
        from galen.models import vgg
        model = vgg.VAE(device=device)
    else:
        raise ValueError("Model must be 'resnet' or 'vgg'!")

    if weights is None:
        weights = restore_latest_weights(dir)

    model.load_state_dict(torch.load(weights, map_location=device))

    model.cpu().eval()
    return model


# Not really used anymore
def restore_latest_weights(dir):
    """
    Find latest weights at given directory

    Args:
    * dir
	Path to directory to search (str)

    Returns:
        Path to latest weights (str)
    """
    checkpoints = sorted(glob.glob(dir + '*.pth'), key=os.path.getmtime)
    if len(checkpoints) > 0:
        return checkpoints[-1]


if __name__ == "__main__":
    print("Done!")
