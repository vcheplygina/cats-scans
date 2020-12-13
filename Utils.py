import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from skimage import io, color
from skimage.filters import gabor, median
from skimage.feature import greycomatrix, greycoprops
import cv2
import os

def batchgenerator(n, upper, lower):
    """
    Creates an array of n unique and random numbers
    returns the array
    """
    img_nums= random.sample(range(lower, upper), n)
    
    return img_nums

def npytoarray(nums, x=112, y=112):
    """
    Converts npy file to numpy array with of images
    n = amount of images added to array
    x,y = amount of pixels in x and y directions
    returns the array
    """
    pics = np.load('all_imgs.npy', allow_pickle=True)

    img_list = []
    for img in pics[nums]:
        #Choose to resize image if preferred
        img = cv2.resize(img, (x,y))
        img_list.append(img)
    img_array = np.array(img_list)

    return img_array

def preprocesser(input_array):
    """
    Converts input array of images to greyscale and median filtered arrays
    Converts greyscaled image values between [0, 1] to uint8 type with [0, 256]
    returns array of greyscale images and median filtered greyscale images.
    """
    original_imgs = []
    original_grey_imgs = []
    filtered_imgs = []

    for x in range(len(input_array)):
        original_imgs.append(input_array[x])
        original = input_array[x]
        original_grey = color.rgb2gray(original)
        grey_filtered = median(original_grey)

        original_grey = original_grey*256
        grey_filtered = grey_filtered*256

        original_grey = original_grey.astype(np.uint8)
        grey_filtered = grey_filtered.astype(np.uint8)

        original_grey_imgs.append(original_grey)
        filtered_imgs.append(grey_filtered)

    return original_grey_imgs, filtered_imgs

def visualizer(nrows, ncols, nums, original_imgs, original_grey_imgs, filtered_imgs):
    fig, ax = plt.subplots(nrows, ncols)

    for x in range(ncols):
        ax[0,x].imshow(original_imgs[x])
        ax[0,x].set_title("Original image {}".format(nums[x]))

    for x in range(ncols):
        ax[1,x].imshow(original_grey_imgs[x], cmap=plt.cm.gray)
        ax[1,x].set_title("Grey image {}".format(nums[x]))
    
    for x in range(ncols):
        ax[2,x].imshow(filtered_imgs[x], cmap=plt.cm.gray)
        ax[2,x].set_title("Filtered grey image {}".format(nums[x]))
    
    fig.tight_layout()
    plt.show()

def computeglcm(original_imgs, original_grey, grey_filtered, d, a, levels):
    glcms = []
    contrast = []
    dissimilarity = []
    homogeneity = []
    asm = []
    energy = []
    correlation = []

    for x in range(len(original_imgs)):
        glcm = greycomatrix(original_grey[x], distances=d, angles=a, levels=levels)
        glcms.append(glcm)
        contrast.append(greycoprops(glcm, 'contrast')[0,0])
        dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0,0])
        homogeneity.append(greycoprops(glcm, 'homogeneity')[0,0])
        asm.append(greycoprops(glcm, 'ASM')[0,0])
        energy.append(greycoprops(glcm, 'energy')[0,0])
        correlation.append(greycoprops(glcm, 'correlation')[0,0])

    print('contrast:',contrast)
    print('dissimilarity:',dissimilarity)
    print('homogeneity:',homogeneity)
    print('asm:',asm)
    print('energy:',energy)
    print('correlation:',correlation)

x = batchgenerator(4, 8060, 0)

array = npytoarray(nums=x)

grey, filtered = preprocesser(array)

computeglcm(original_imgs=array, original_grey=grey, grey_filtered=filtered, d=[25], a=[45], levels=256)

visualizer(nrows=3, ncols=4, nums=x, original_imgs=array, original_grey_imgs=grey, filtered_imgs=filtered)