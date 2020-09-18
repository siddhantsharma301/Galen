"""
Linear Interpolation Script for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import os
import cv2
import torch
import errno
import argparse
import numpy as np
from torchvision.utils import save_image
from galen.utils import image as image_utils
from galen.utils import model as model_utils


def interpolate(img1, img2, model):
    """
    Linear interpolation between two images

    Args:
    * img1
        image (`torch.Tensor`)
    * img2
        image (`torch.Tensor`)
    * model
        model to use for interpolate (`torch.nn.Module`)

    Returns:
    10 images of interpolation between given images
    """
    model.eval()
    mu1, logvar1 = model.encode(img1)
    z1 = model.reparametrize(mu1, logvar1)
    mu2, logvar2 = model.encode(img2)
    z2 = model.reparametrize(mu2, logvar2)

    factors = np.linspace(1, 0, num=10)
    res = []

    with torch.no_grad():
        for f in factors:
            z = (f * z1 + (1 - f) * z2)
            im = torch.squeeze(model.decode(z))
            res.append(im)
    return res

def model_to_resize_dims(model):
    """
    Return resizing dimensions based on model type

    Args:
    * model
    Type of model

    Returns:
        Resize dimensions for given model type
    """
    if model == "resnet":
        return (224, 224)
    elif model == "vgg":
        return (128, 128)
    else:
        raise ValueError("Model must be ResNet or VGG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Linear Inteprolation between two images")
    parser.add_argument('--image1', required = True, help = 'Image 1')
    parser.add_argument('--image2', required = True, help = 'Image 2')
    parser.add_argument('--model', required = True, choices = ['resnet', 'vgg'], help = 'Model type to use')
    parser.add_argument('--dir', required = False, help = 'Path to look for weights for given model')
    parser.add_argument('--weights', required = False, help = 'Path to weights')
    parser.add_argument('--save', required = False, default = 'galen/outputs/',help = 'Dir to save interpolation')
    args = vars(parser.parse_args())

    if args['dir'] is not None and args['weights'] is not None:
    	raise ValueError("Both directory and weights cannot be given at same time")
    elif args['dir'] is None and args['weights'] is None:
        raise ValueError("Need to provide a directory or weights")

    if not os.path.exists(args['image1']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    if not os.path.exists(args['image2']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    print("Reading images...")
    resize_dims = model_to_resize_dims(args['model'])
    img1 = cv2.imread(args['image1'])
    img1 = image_utils.img_to_tensor(img1, resize_dims=resize_dims)
    img2 = cv2.imread(args['image2'])
    img2 = image_utils.img_to_tensor(img2, resize_dims=resize_dims)

    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_utils.load_model(device, model_name=args['model'], weights=args['weights'], dir=args['dir'])

    print("Interpolating...")
    inter = interpolate(img1, img2, model)

    print("Done! Saving image...")
    save_image(inter, args['save'] + 'interpolation.png', padding=0, nrow=10)
    print("Saved!")
