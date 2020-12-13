import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from skimage import io, color
from skimage.filters import gabor, median
from skimage.feature import greycomatrix, greycoprops
import cv2
import os
import math

def batchgenerator(n, upper, lower):
    """
    Creates an array of n unique and random numbers
    returns the array

    Used to determine the amount and the indexes of images used for analysis
    """
    img_nums= random.sample(range(lower, upper), n)
    
    return img_nums

def bintoarray(path_to_data, nums, x=96, y=96):
    """
    param path_to_data: the file containing the binary images from the STL-10 dataset
    x,y = amount of pixels in x and y directions
    return: an array containing all the images

    Used to extract images from Stl-10 dataset
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, x, y))

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

def npytoarray(data ,nums, x=112, y=112):
    """
    Converts npy file to numpy array with of images
    n = amount of images added to array
    x,y = amount of pixels in x and y directions
    returns the array

    Used to extract images from Sti-10 dataset
    """
    pics = np.load('all_imgs_{}.npy'.format(data), allow_pickle=True)

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
    Conversion is done with values Y = 0.2125 R + 0.7154 G + 0.0721 B
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
    """
    Plots a matrix with the original pictures in the first row, the grey scaled on the second row and grey median filtered pictures on the third
    nrows = number of rows
    ncols = number of columns, also the number of unique pictures shown
    """

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
    """
    First computes the grey level co-occurence matrix of the grey scaled images.
    Then from this matrix, computes different statistics and stores them separate vectors for each feature
    These features are: contrast, dissimilarity, homogeneity, asm, energy and correlation
    """
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

    return contrast, dissimilarity, homogeneity, asm, energy, correlation

def createvector(name, nums, contrast, dissimilarity, homogeneity, asm, energy, correlation):
    """
    Creates a single feature vector for each image
    Stores the feature values of each image vector in a .txt file
    Creates a 'batchvector' that contains the average feature values of the whole batch
    """
    vectorfile = open("vectors {}.txt".format(name), "w+")
    vectorfile.write("Contrast, dissimilarity, homogeneity, asm, energy, correlation \n")
    batchvector = []
    for x in range(len(nums)):
        vector = []
        vector.append(contrast[x])
        vector.append(dissimilarity[x])
        vector.append(homogeneity[x])
        vector.append(asm[x])
        vector.append(energy[x])
        vector.append(correlation[x])
        vectorfile.write("Vector number {} statistics: {} \n".format(nums[x], vector))
        
    batchvector = [sum(contrast)/len(nums), sum(dissimilarity)/len(nums), sum(homogeneity)/len(nums), sum(asm)/len(nums), sum(energy)/len(nums), sum(correlation)/len(nums)]
    vectorfile.write("The average statistics of this batch \n")
    vectorfile.write("Contrast:{}, Dissimilarity:{}, Homogeneity:{}, Asm:{}, Energy:{}, Correlation:{}".format(batchvector[0], batchvector[1], batchvector[2], batchvector[3], batchvector[4], batchvector[5]))

    print("The average statistics of this batch")
    print("Contrast:{}, Dissimilarity:{}, Homogeneity:{}, Asm:{}, Energy:{}, Correlation:{}".format(batchvector[0], batchvector[1], batchvector[2], batchvector[3], batchvector[4], batchvector[5]))

    vectorfile.close()
    return batchvector

# Comparisons for vectors, via: https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity

def compareeucdistance(a, b):
    """
    Computes the euclidian distance between two vectors a and b
    """
    eucdist = 0
    for x in range(len(a)):
        partx = (a[x]-b[x])**2
        eucdist = eucdist + partx
    eucdist = math.sqrt(eucdist)
    print("The euclidian distance between these vectors is", eucdist)

    return eucdist 

def comparecosine(a, b):
    """
    Computes the cosine of the angle theta between two vectors a and b
    """
    a = np.array(a)
    b = np.array(b)
    
    upper = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    lower = norma*normb 

    cosine = upper/lower
    print("The Cosine of angle Theta between these vectors is", cosine)

    return cosine

def normalize(a, b):
    """
    Normalizes two vectors a and b to values between 0 and 1
    The highest value between the two vectors for each feature is taken as 1
    """
    max = 0
    maxa = np.amax(a)
    maxb = np.amax(b)

    if maxa > maxb:
        max = maxa
    else:
        max = maxb
    
    a = a/max
    b = b/max

    return a, b



