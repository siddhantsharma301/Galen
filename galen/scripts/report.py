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


if __name__ == "__main__":
    # Set up threshold values
    SURF_THRESH = 11.5
    KAZE_THRESH = 28.5

    # Set up CLI parser
    parser = argparse.ArgumentParser(description = 'Diagnose chest x-ray; either healthy or unhealthy')
    parser.add_argument('--image', required = True, help = 'Image to diagnose')
    parser.add_argument('--weights', required = False, help = 'Path to weights')
    parser.add_argument('--dir', required = False, default = './training/checkpoints/', help = 'Path to dir for weights')
    args = vars(parser.parse_args())
    filename = args['image']
    
    # Check if image given exists
    if not os.path.exists(args['image']):
        raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filename)

    if args['weights'] is not None and args['dir'] is not None:
    	raise ValueError("Both weights and directory cannot be given at same time")
    
    # Start timing
    start = time.time()

    # Read and convert image to tensor
    print("Reading image...")
    img = cv2.imread(args['image'])
    img = image_utils.img_to_tensor(img)

    # Load model
    print("Loading model...")
    model = model_utils.load_model(weights=args['weights'], dir=args['dir'])

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


    
    
