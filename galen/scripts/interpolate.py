"""
Linear Interpolation Script for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import os
import cv2
import torch
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
    z1 = model.encode(img1)
    z2 = model.encode(img2)

    factors = np.linspace(1, 0, num=10)
    res = []

    with torch.no_grad():
        for f in factors:
            z = (f * z1 + (1 - f) * z2)
            im = torch.squeeze(model.decode(z))
            res.append(im)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Linear Inteprolation between two images")
    parser.add_argument('--image1', required = True, help = 'Image 1')
    parser.add_argument('--image2', required = True, help = 'Image 2')
    parser.add_argument('--weights', required = False, help = 'Path to weights')
    parser.add_argument('--dir', required = False, default = './training/checkpoints', help = 'Dir to find latest weights')
    parser.add_argument('--save', required = False, default = './outputs/',help = 'Dir to save interpolation')
    args = vars(parser.parse_args())

    if args['weights'] is not None and args['dir'] is not None:
    	raise ValueError("Both cannot be weights and dir cannot be given at same time")

    if not os.path.exists(args['image1']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    if not os.path.exists(args['image2']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    print("Reading images...")
    img1 = cv2.imread(args['image1'])
    img1 = image_utils.img_to_tensor(img1)
    img2 = cv2.imread(args['image2'])
    img2 = image_utils.img_to_tensor(img2)

    print("Loading model...")
    model = model_utils.load_model(weights=args['weights'], dir=args['dir'])

    print("Interpolating...")
    inter = interpolate(img1, img2, model)

    print("Done! Saving image...")
    save_image(inter, args['save'] + 'interpolation.png', padding=0, nrow=10)
    print("Saved!")
