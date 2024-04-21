import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image

    """
    # Get original image shape
    height, width = img.shape

    # Calculate padding size
    pad_size = filter_size // 2

    # Create padded image with zeros
    padded_img = np.zeros((height + 2 * pad_size, width + 2 * pad_size), dtype=img.dtype)

    # Copy original image into padded image
    padded_img[pad_size:pad_size + height, pad_size:pad_size + width] = img
    return padded_img
  # Need to implement here


def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    # Get dimensions of the original image
    height, width = img.shape

    # Create a padded image
    padded_img = padding_img(img, filter_size)

    # Initialize smoothed image
    smoothed_img = np.zeros_like(img)

    # Apply mean filter
    for i in range(height):
        for j in range(width):
            # Extract the region of interest
            roi = padded_img[i:i+filter_size, j:j+filter_size]
            # Calculate the mean value
            smoothed_img[i, j] = np.mean(roi)

    return smoothed_img

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
  # Need to implement here
    height, width = img.shape

    # Create a padded image
    padded_img = padding_img(img, filter_size)

    # Initialize smoothed image
    smoothed_img = np.zeros_like(img)

    # Apply median filter
    for i in range(height):
        for j in range(width):
            # Extract the region of interest
            roi = padded_img[i:i+filter_size, j:j+filter_size]
            # Calculate the median value
            smoothed_img[i, j] = np.median(roi)

    return smoothed_img

def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Ensure both images have the same data type
    gt_img = gt_img.astype(np.float64)
    smooth_img = smooth_img.astype(np.float64)

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((gt_img - smooth_img) ** 2)

    # Calculate maximum pixel value
    max_pixel = np.max(gt_img)

    # Calculate PSNR (Peak Signal to Noise Ratio)
    psnr_score = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr_score
    # Need to implement here

def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise_path = "/content/noise.png"
    img_gt_path = "/content/ori_img.png"
    img_noise = read_img(img_noise_path)
    img_gt = read_img(img_gt_path)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img_noise, filter_size)
    show_res(img_noise, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img_gt, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img_noise, filter_size)
    show_res(img_noise, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img_gt, median_smoothed_img))

