"""
Report Generation Script for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import os
import cv2
import time
import errno
import torch
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.FATAL)
from galen.utils import image as image_utils
from galen.utils import model as model_utils
from galen.utils import plot as plot_utils


"""
Helper functions for report generation
"""
def create_parser():
    """
    Helper to create an argument parser
    """
    parser = argparse.ArgumentParser(description = 'Diagnose chest x-ray; either healthy or unhealthy')
    parser.add_argument('--image', required = True, help = 'Image to diagnose')
    parser.add_argument('--model', required = True, choices = ['resnet', 'vgg'], help = 'Model type to use')
    parser.add_argument('--dir', required = False, default = './training/checkpoints/', help = 'Path to look for weights for given model')
    parser.add_argument('--weights', required = False, help = 'Path to weights')
    parser.add_argument('--cuda', required = False, default = 'cpu', help = 'Use CUDA acceleration')
    return parser


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
    # Set up threshold values
    SURF_THRESH = 11.5
    KAZE_THRESH = 28.5

    # Set up CLI parser
    parser = create_parser()
    args = vars(parser.parse_args())
    filename = args['image']
    
    if not os.path.exists(args['image']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    elif args['dir'] is not None and args['weights'] is not None:
    	raise ValueError("Both directory and weights cannot be given at same time")
    elif args['dir'] is None and args['weights'] is None:
        raise ValueError("Need to provide a directory or weights")
    
    # Start timing
    start = time.time()

    # Read and convert image to tensor and resize image
    print("Reading image...")
    img = cv2.imread(args['image'])
    resize_dims = model_to_resize_dims(args['model'])
    img = image_utils.img_to_tensor(img, resize_dims=resize_dims)

    # Load model
    print("Loading model...")
    model = model_utils.load_model(device=args['cuda'], model=args['model'], weights=args['weights'], dir=args['dir'])

    # Reconstruct image, convert to tensor, and post-process with blurring
    print("Reconstructing...")
    recon, _, _ = model(img)
    img = image_utils.tensor_to_img(img)
    recon = image_utils.tensor_to_img(recon)
    recon = cv2.bilateralFilter(recon, 1, 3, 3)

    # Get difference heatmap
    diff = image_utils.get_heatmap(img, recon)

    # Score the images using SURF and KAZE
    print("Scoring image...")
    surf_score = image_utils.surf_match(np.copy(img), np.copy(recon))
    kaze_score = image_utils.kaze_match(np.copy(img), np.copy(recon))

    # Check if scores pass threshold
    surf_healthy = True if surf_score >= SURF_THRESH else False
    kaze_healthy = True if kaze_score >= KAZE_THRESH else False
    # Both SURF and KAZE must indicate the image is healthy
    healthy = True if (surf_healthy + kaze_healthy) == 2 else False

    # Plot report
    if healthy:
        plot_utils.plot_report(img, recon, img, diff)
    else:
        bbox = image_utils.get_bbox(img, recon)
        plot_utils.plot_report(img, recon, bbox, diff)

    # Save some outputs for logging 
    cv2.imwrite('./outputs/recon.png', recon * 255.)
    cv2.imwrite('./outputs/diff.png', diff * 255.)

    # Print report to command line
    print('----------------------------------------------------------')
    print('The image is {}'.format('healthy.' if healthy else 'unhealthy.'))
    print('The SURF score is: {}'.format(round(surf_score, 3)) + ".")
    if surf_healthy:
        print("The SURF score indicate the image is healthy.")
    else:
        print("The SURF score indicate the image is unhealthy.")
    print('The KAZE score is: {}'.format(round(kaze_score, 3)) + ".")
    if kaze_healthy:
        print("The KAZE score indicate the image is healthy.")
    else:
        print("The KAZE score indicate the image is unhealthy.")
    print("Time taken: {} seconds".format(time.time() - start))
    print('----------------------------------------------------------')
