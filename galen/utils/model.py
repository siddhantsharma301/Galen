"""
Model Utilities for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import os
import glob
import torch
from galen.models import vgg, resnet


def load_model(device, model_name, weights=None, dir=None):
    """
    Load a trained model from indicated weights or dir

    Args:
    * device
        Indicate what device to use (str)
        For CUDA acceleration 
    * model_name
        Type of model to use (str)
    * weights
    	OPTIONAL
    	Path to weights (str)
    	Default: None
    * dir
    	OPTIONAL
    	Path to dir to find weights (str)
    	Default: None
    

    Returns:
    Model with loaded weights (`torch.nn.module`)
    """
    if model_name == "resnet":
        model = resnet.VAE(device=device)
    elif model_name == "vgg":
        model = vgg.VAE(device=device)
    else:
        raise ValueError("Need to select between ResNet and VGG Galen")

    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        latest_weights = restore_latest_weights(model_name, dir)
        model.load_state_dict(torch.load(latest_weights, map_location=device))
    
    model.eval()
    return model


def restore_latest_weights(model, dir):
    """
    Args:
    * model
    Type of model to get weights for (str)
    * dir
	Path to directory to search (str)

    Returns:
        Path to latest weights (str)
    """
    # TODO: FIX THIS TO WORK FOR DISCRIMINATOR MODELS TOO
    path = dir + model + '.vae.pth'
    checkpoints = sorted(glob.glob(path), key=os.path.getmtime)
    if len(checkpoints) > 0:
        return checkpoints[-1]

if __name__ == "__main__":
    print("Done!")
