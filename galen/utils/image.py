"""
Image Utilities for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import cv2
import torch
import numpy as np
from skimage import measure


def to_channels_last(img):
    """
    Turn (H, W, C) to (C, H, W)

    Args:
    * img 
        image (`np.ndarray`)

    Returns:
    channels first (`np.ndarray`)
    """
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img


def to_channels_first(img):
    """
    Turn (C, H, W) to (H, W, C)

    Args:
    * img 
        image (`np.ndarray`)

    Returns:
    channels last image (`np.ndarray`)
    """
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img


def img_to_tensor(img, resize=True, clahe=True):
    """
    Convert image to Pytorch Tensor

    Args:
    * img
        Image to convert to Pytorch Tensor
        NOTE: Assume `np.uint8` input dtype
    * resize
        OPTIONAL
        Resize image to 128x128 (bool)
        Default: True
    * clahe
        OPTIONAL 
        Apply CLAHE to image (bool)
        Default: True

    Returns:
    Pytorch image Tensor
    """
    if resize:
        img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
    if clahe:
        img = clahe_image(img)
    img = img.astype('float32')
    img /= 255.
    img = to_channels_last(img)
    return torch.from_numpy(img[np.newaxis]).float()


def tensor_to_img(tensor):
    """
    Convert Pytorch Tensor to Numpy image

    Args:
    * tensor
        Pytorch Tensor to convert (`torch.Tensor`)

    Returns:
    Numpy image (`np.ndarray`)
    """
    tensor = tensor.data.numpy()
    tensor = tensor[0]
    return to_channels_first(tensor)


def clahe_image(img):
    """
    CLAHE-ify an image

    Args:
    * img 
        image (`np.ndarray`)

    Returns:
    CLAHE-ified image (`np.ndarray`)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8, 8))
    lab_planes[0]= clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def surf_match(img1, img2, ratio=0.5, save=True):
    """
    Return SURF matching score between two images
    First image should be original/input image, 
    second image should be reconstruction image
    NOTE: Assume all images are `np.float32`
    TODO: Find a better way to save SURF output,
          it's not pretty right now and hard to 
          understand

    Args:
    * img1
        image (`np.ndarray`)
    * img2
        image (`np.ndarray`)
    * ratio
        OPTIONAL
        How accurate of a match (int)
        The lower the ratio, the more accurate the match
    * save
        OPTIONAL
        Save visual output of SURF or not (bool)
        Default: True

    Returns:
    SURF similarity score (float)
    """
    # Convert images to `np.uint8`
    img1 *= 255.
    img1 = img1.astype('uint8')
    img2 *= 255.
    img2 = img2.astype('uint8')

    # Create SURF descriptor and find keypoints in both images
    sift = cv2.xfeatures2d.SURF_create()
    kp_1, desc_1 = sift.detectAndCompute(img1, None)
    kp_2, desc_2 = sift.detectAndCompute(img2, None)

    # Search the keypoints for matches
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    # Find matches (within some give or take)
    good_points = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_points.append(m)

    # Save the output
    if save:
        cv2.imwrite('./outputs/surf.png', cv2.drawMatches(img1, kp_1, img2, kp_2, good_points, None))

    # Return the score
    return len(good_points) * 100 / len(kp_2)


def kaze_match(img1, img2, ratio=0.4, save=True):
    """
    Return KAZE matching score between two images
    First image should be original/input image, 
    second image should be reconstruction image
    NOTE: Assume all images are `np.float32`
    TODO: Find a better way to save KAZE output,
          it's not pretty right now and hard to 
          understand

    Args:
    * img1
        image (`np.ndarray`)
    * img2
        image (`np.ndarray`)
    * ratio
        OPTIONAL
        How accurate of a match (int)
        The lower the ratio, the more accurate the match
    * save
        OPTIONAL
        Save visual output of KAZE or not (bool)
        Default: True

    Returns:
    KAZE similarity score (float)
    """
    # Convert images to `np.uint8`
    img1 *= 255.
    img1 = img1.astype('uint8')
    img2 *= 255.
    img2 = img2.astype('uint8')

    # Create KAZE descriptor and find keypoints in both images
    kaze = cv2.KAZE_create()
    kp_1, desc_1 = kaze.detectAndCompute(img1, None)
    kp_2, desc_2 = kaze.detectAndCompute(img2, None)

    # Search the keypoints for matches
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches (within some give or take)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good_points.append(m)

    # Save the output
    if save:
        cv2.imwrite('./outputs/kaze.png', cv2.drawMatches(img1, kp_1, img2, kp_2, good_points, None))

    # Return the score
    return len(good_points) * 100 / len(kp_2)


def get_bbox(img1, img2):
    """
    Get the area of the largest difference between the original
    and reconstructions
    Only should be used if input image is unhealthy

    Args:
    * img1
        original image (`np.ndarray`)
    * img2
        reconstruction image (`np.ndarray`)

    Returns:
    bbox image (`np.ndarray`)
    """
    # Grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Get differences using SSIM
    (score, diff) = measure.compare_ssim(img1_gray, img2_gray, full=True)
    diff = (diff * 255).astype("uint8")
    # Threshold differences 
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Extract contours
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    # Find the largest contour (to a certain extent)
    # Some limits bc contour shouldn't be whole image
    largest = cv2.contourArea(contours[0])
    largest_idx = 0
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area > 40 and area < 6000:
            if largest < area:
                largest = area
                largest_idx = c
    img = np.copy(img1)
    x,y,w,h = cv2.boundingRect(contours[largest_idx])
    # Place contour bbox
    cv2.rectangle(img, (x, y), (x + w - 2, y + h - 2), (255,0,0), 2)
    return img


def get_heatmap(img1, img2):
    """
    Get the RGB difference between original and reconstruction
    First image should be original/input image
    Second image should be reconstruction image

    Args:
    * img1
        image (`np.ndarray`)
    * img2
        image (`np.ndarray`)

    Returns:
    heatmap image (`np.ndarray`)
    """
    error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
    error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
    error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))
    return np.maximum(np.maximum(error_r, error_g), error_b)
