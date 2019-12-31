"""
Model Utilities for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import os
import glob
import torch
from galen.models import vae


def load_model(weights=None, dir='./training/checkpoints/'):
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
    model = vae.VAE(512)
    if weights is None:
    	weights = restore_latest_weights(dir)
    model.load_state_dict(torch.load(weights, map_location={'cuda:0': 'cpu'}))
    model.cpu().eval()
    return model


def restore_latest_weights(dir):
    """
    Find latest weights at given directory

    Args:
    * dir
	Path to directory to search (str)

    Returns:
        Path to latest weights (str)
    """
    checkpoints = sorted(glob.glob(dir + '*.pt'), key=os.path.getmtime)
    if len(checkpoints) > 0:
        return checkpoints[-1]

if __name__ == "__main__":
    print("Done!")
