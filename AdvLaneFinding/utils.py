"""
    Utility functions module
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
import numpy as np

def grayscale(img):
    """
        Returns an image grayscaled.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb2hls(img):
    """
        Converts to HLS color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def rgb2hsv(img):
    """
        Converts to HSV color space.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def select_channel(img, channel='s'):
    """
        Select a channel from a HLS image.
    """
    if channel == 'h':
        ich = 0
    elif channel == 'l':
        ich = 1
    elif channel == 's':
        ich = 2
    return img[:, :, ich]

def extract_white_yellow(img):
    """
        Extracts white and yellow pixels.
    """
    hsv = rgb2hsv(img)
    l_yellow = np.array([0, 100, 100], dtype=np.uint8)
    u_yellow = np.array([190, 250, 255], dtype=np.uint8)
    l_white = np.array([255, 255, 255], dtype=np.uint8)
    u_white =np.array([200, 200, 200], dtype=np.uint8)
    white = cv2.inRange(img, u_white, l_white)
    yellow = cv2.inRange(hsv, l_yellow, u_yellow)
    return cv2.bitwise_or(white, yellow)

def gaussian_blur(img, kernel_size=5):
    """
        Applies a Gaussian Noise kernel
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def get_region(image):
    """
        Returns the part which is likely to contain lane lines.
    """
    bottom_left = (image.shape[1] * .15, image.shape[0])
    bottom_right = (image.shape[1] * .93, image.shape[0])
    top_left = (image.shape[1] * .4, image.shape[0] / 2)
    top_right = (image.shape[1] * .6, image.shape[0] / 2)
    white = np.zeros_like(image)
    points = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(white, points, 255)
    return cv2.bitwise_and(image, white)

def remove_atypical(x, y, perc=5, horitzontal=True):
    """
        Removes atypical values based on percentile.
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    if horitzontal:
        lower_bound = np.percentile(x, perc)
        upper_bound = np.percentile(x, 100 - perc)
        selection = (x >= lower_bound) & (x <= upper_bound)
    else:
        lower_bound = np.percentile(y, perc)
        upper_bound = np.percentile(y, 100 - perc)
        selection = (y >= lower_bound) & (y <= upper_bound)
    return x[selection], y[selection]

def write_image(img, path):
    """
        Writes an image into a file.
    """
    cv2.imwrite(path, img)

def is_grayscaled(img):
    """
        Returns True if an images is grayscaled
    """
    return not (len(img.shape) == 3 and img.shape[2] == 3)

def plot_points(img, one, two, three, four, name, output='./output_images'):
    """
        Writes a image with four points.
    """
    if is_grayscaled(img):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.plot(one[0], one[1], 'r.')
    plt.plot(two[0], two[1], 'g.')
    plt.plot(three[0], three[1], 'b.')
    plt.plot(four[0], four[1], 'y.')
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close('all')

def plot_histogram(img, histogram, name, output='./output_images'):
    """
        Plots the histogram over the image.
    """
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    f.tight_layout()
    if is_grayscaled(img):
        ax1.imshow(img, cmap='gray')
    else:
        ax1.imshow(img)
    ax2.plot(histogram)
    ax2.set_title('Histogram', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close(f)

def write_two_img(imgs, titles, name, output='./output_images'):
    """
        Writes two images in a single plot.

        Attributes:
            imgs: Array of 2 images, one for each subplot.
            titles: Array of 2 titles, one for each subplot.
            name: Name of the file
            output: Output directory
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if is_grayscaled(imgs[0]):
        ax1.imshow(imgs[0], cmap='gray')
    else:
        ax1.imshow(imgs[0])
    if is_grayscaled(imgs[1]):
        ax2.imshow(imgs[1], cmap='gray')
    else:
        ax2.imshow(imgs[1])
    ax1.set_title(titles[0], fontsize=50)
    ax2.set_title(titles[1], fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join(output, name), bbox_inches='tight')
    plt.close(f)

def read_image(path):
    """
        Reads an image from file.
    """
    return mpimg.imread(path)

def list_dir(path):
    """
        Lists all the files in a directory.
    """
    return glob.glob(os.path.join(path, '*'))
