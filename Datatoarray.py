import glob
import math
import os
import random

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, io
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor, median

#All functions take the dataset, and turn these into arrays of shape (n, 112, 112, 3). n being the number of images chosen in batchgenerator.  

def isic2018toarray(nums):
    img_array = []
    for x in nums:
        image = mpimg.imread(r"D:\BEP\ISIC2018\ISIC2018_Task3_Training_Input\ISIC_00{}.jpg".format(x))
        image_data = np.array(image, dtype='uint8')
        img_resized = cv2.resize(image_data, (112, 112))
        img_array.append(img_resized)

    img_arr = np.array(img_array)
        
    return img_arr

def stl10toarray(nums):
    """
    param path_to_data: the file containing the binary images from the STL-10 dataset
    x,y = amount of pixels in x and y directions
    return: an array containing all the images

    Used to extract images from Stl-10 dataset
    """

    with open('unlabeled_X.bin', 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))

        img_list=[]

        for img in images[nums]:
            img_list.append(img)
        img_array = np.array(img_list)

        return img_array

def sti10toarray(nums):
    """
    Converts npy file to numpy array with of images
    n = amount of images added to array
    x,y = amount of pixels in x and y directions
    returns the array

    Used to extract images from Sti-10 dataset
    """
    pics = np.load('all_imgs_sti10.npy', allow_pickle=True)

    img_list = []
    for img in pics[nums]:
        #Choose to resize image if preferred
        img = cv2.resize(img, (112,112))
        img_list.append(img)
    img_array = np.array(img_list)

    return img_array

def imagenettoarray(nums):
    pics = np.load(r'D:\BEP\test\all_imgs.npy', allow_pickle=True)

    img_list = []
    for img in pics[nums]:
        #Choose to resize image if preferred
        img = cv2.resize(img, (112,112))
        img_list.append(img)
    img_array = np.array(img_list)

    return img_array

def dtdtoarray(nums):
    img_array = []
    path = glob.glob(r"D:\BEP\dtd\moddedimages\*.jpg")
    for img in path:
        image = cv2.imread(img)
        image = cv2.resize(image, (112,112))
        img_array.append(image)
    final_array = np.array(img_array)

    final_array_nums = []
    for x in nums:
        final_array_nums.append(final_array[x])

    final_array_nums = np.array(final_array_nums)

    return final_array_nums

def kimiatoarray(nums):
    img_array = []
    #path = glob.glob(r"C:\Users\s166646\Downloads\BEP\kimia_path_960\*.tif")
    path = glob.glob(r"D:\BEP\kimia_path_960\*.tif")
    for img in path:
        image = cv2.imread(img)
        image = cv2.resize(image, (112,112))
        img_array.append(image)
    final_array = np.array(img_array)

    final_array_nums = []
    for x in nums:
        final_array_nums.append(final_array[x])

    final_array_nums = np.array(final_array_nums)

    return final_array_nums

def pcamtoarray(nums):
    img_array = []
    path = glob.glob(r"D:\BEP\PCam\png_images\*.png")
    for x in nums:
        image = cv2.imread(path[x])
        image = cv2.resize(image, (112,112))
        img_array.append(image)
    final_array = np.array(img_array)

    #final_array_nums = []
    #for x in nums:
    #    final_array_nums.append(final_array[x])

    #final_array_nums = np.array(final_array_nums)

    return final_array

def chestxraytoarray(nums):
    img_array = []
    path = glob.glob(r"D:\BEP\chest_xray\all_images\*.jpeg")
    for img in path:
        image = cv2.imread(img)
        image = cv2.resize(image, (112,112))
        img_array.append(image)
    final_array = np.array(img_array)

    final_array_nums = []
    for x in nums:
        final_array_nums.append(final_array[x])

    final_array_nums = np.array(final_array_nums)

    return final_array_nums
