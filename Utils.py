import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
from PIL import Image
from skimage import io, color
from skimage.filters import gabor, median
from skimage.feature import greycomatrix, greycoprops
import cv2
import os
import glob
import math

def batchgenerator(n, upper, lower):
    """
    Creates an array of n unique and random numbers
    returns the array

    Used to determine the amount and the indexes of images used for analysis
    """
    img_nums= random.sample(range(lower, upper), n)
    
    return img_nums

def isic2018toarray(nums):
    img_array = []
    for x in nums:
        image = mpimg.imread(r"D:\BEP\ISIC2018\ISIC2018_Task3_Training_Input\ISIC_00{}.jpg".format(x))
        image_data = np.array(image, dtype='uint8')
        img_resized = cv2.resize(image_data, (112, 112))
        img_array.append(img_resized)

    img_arr = np.array(img_array)
        
    return img_arr

def stl10toarray(path_to_data, nums, x=96, y=96):
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

def sti10toarray(data ,nums, x=112, y=112):
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

def batchvector(nums, contrast, dissimilarity, homogeneity, asm, energy, correlation):
    batchvector = []
    batchvector = [sum(contrast)/len(nums), sum(dissimilarity)/len(nums), sum(homogeneity)/len(nums), sum(asm)/len(nums), sum(energy)/len(nums), sum(correlation)/len(nums)]
    return batchvector

def writevector(name, nums, contrast, dissimilarity, homogeneity, asm, energy, correlation):
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

def resultvisualizer(a):
    
    x = np.array([0.9, 0.907, 0.908, 0.911, 0.912])
    y = np.array(a)

    plt.scatter(x, y)
    plt.show()

def sevenplotvisualizer(contrast, dissimilarity, homogeneity, asm, energy, correlation, batchvector, source):
    fig, ax = plt.subplots(3, 3)

    fig.suptitle("Difference between {} and other sets".format(source))

    x = np.array([0.9, 0.907, 0.908, 0.912, 0.928, 0.946])
    
    ax[0,0].scatter(x, contrast)
    ax[0,0].set_title("Contrast")

    ax[0,1].scatter(x, dissimilarity)
    ax[0,1].set_title("Dissimilarity")

    ax[0,2].scatter(x, homogeneity)
    ax[0,2].set_title("Homogeneity")

    ax[1,0].scatter(x, asm)
    ax[1,0].set_title("Asm")

    ax[1,1].scatter(x, energy)
    ax[1,1].set_title("Cnergy")

    ax[1,2].scatter(x, correlation)
    ax[1,2].set_title("Correlation")

    ax[2,1].scatter(x, batchvector)
    ax[2,1].set_title("Euclidian distance")

    plt.show()

def zeromean(a, b, c, d, e, f, g, h):
    ab = np.concatenate([a, b])
    abc = np.concatenate([ab, c])
    abcd = np.concatenate([abc, d])
    abcde = np.concatenate([abcd, e])
    abcdef = np.concatenate([abcde, f])
    abcdefg = np.concatenate([abcdef, g])
    abcdefgh = np.concatenate([abcdefg, h]) 

    mean = np.mean(abcdefgh)
    print('mean is:{}'.format(mean))
    std = np.std(abcdefgh)
    print('std is:{}'.format(std))
    
    zeromeanabc = (abcdefgh - mean)/std
    #print(zeromeanabc)

    zeromeanabc = np.split(zeromeanabc, 8)

    a = zeromeanabc[0]
    b = zeromeanabc[1]
    c = zeromeanabc[2]
    d = zeromeanabc[3]
    e = zeromeanabc[4]
    f = zeromeanabc[5]
    g = zeromeanabc[6]
    h = zeromeanabc[7]
    

    return a, b, c, d, e, f, g, h

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

def differencewriter(isicbatchvector, stl10batchvector, dtdbatchvector, sti10batchvector, chestxraybatchvector, pcambatchvector, imagenetbatchvector, kimiabatchvector):
    vectorfile = open("statistics_experiment_1", "w+")
    vectorfile.write("ISIC 2018 statistics \n")
    vectorfile.write("stl10, dtd, sti10, chestxray, pcam, imagenet \n")
    
    vectorfile.write("contrasts for isic \n")
    isicstl10dif = stl10batchvector[0]
    isicdtddif = dtdbatchvector[0]
    isicsti10dif = sti10batchvector[0]
    isicchestxraydif = chestxraybatchvector[0]
    isicpcamdif = pcambatchvector[0]
    isicimgdif = imagenetbatchvector[0]
    isiccontrast = isicbatchvector[0]

    vectorfile.write("{} \n".format(isicstl10dif))
    vectorfile.write("{} \n".format(isicdtddif))
    vectorfile.write("{} \n".format(isicsti10dif))
    vectorfile.write("{} \n".format(isicchestxraydif))
    vectorfile.write("{} \n".format(isicpcamdif))
    vectorfile.write("{} \n".format(isicimgdif))

    vectorfile.write("contrasts for chest\n")
    vectorfile.write("{} \n".format(isicbatchvector[0]))
    vectorfile.write("{} \n".format(sti10batchvector[0]))
    vectorfile.write("{} \n".format(pcambatchvector[0]))
    vectorfile.write("{} \n".format(stl10batchvector[0]))
    vectorfile.write("{} \n".format(dtdbatchvector[0]))
    vectorfile.write("{} \n".format(imagenetbatchvector[0]))

    vectorfile.write("contrasts for pcam \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[0]))
    vectorfile.write("{} \n".format(isicbatchvector[0]))
    vectorfile.write("{} \n".format(sti10batchvector[0]))
    vectorfile.write("{} \n".format(kimiabatchvector[0]))
    vectorfile.write("{} \n".format(stl10batchvector[0]))
    vectorfile.write("{} \n".format(dtdbatchvector[0]))
    vectorfile.write("{} \n".format(imagenetbatchvector[0]))

    vectorfile.write("dissimilarity for isic \n")
    vectorfile.write("{} \n".format(stl10batchvector[1]))
    vectorfile.write("{} \n".format(dtdbatchvector[1]))
    vectorfile.write("{} \n".format(sti10batchvector[1]))
    vectorfile.write("{} \n".format(chestxraybatchvector[1]))
    vectorfile.write("{} \n".format(pcambatchvector[1]))
    vectorfile.write("{} \n".format(imagenetbatchvector[1]))

    vectorfile.write("dissimilarity for chest\n")
    vectorfile.write("{} \n".format(isicbatchvector[1]))
    vectorfile.write("{} \n".format(sti10batchvector[1]))
    vectorfile.write("{} \n".format(pcambatchvector[1]))
    vectorfile.write("{} \n".format(stl10batchvector[1]))
    vectorfile.write("{} \n".format(dtdbatchvector[1]))
    vectorfile.write("{} \n".format(imagenetbatchvector[1]))

    vectorfile.write("dissimilarity for pcam \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[1]))
    vectorfile.write("{} \n".format(isicbatchvector[1]))
    vectorfile.write("{} \n".format(sti10batchvector[1]))
    vectorfile.write("{} \n".format(kimiabatchvector[1]))
    vectorfile.write("{} \n".format(stl10batchvector[1]))
    vectorfile.write("{} \n".format(dtdbatchvector[1]))
    vectorfile.write("{} \n".format(imagenetbatchvector[1]))

    vectorfile.write("homogeneity for isic \n")
    vectorfile.write("{} \n".format(stl10batchvector[2]))
    vectorfile.write("{} \n".format(dtdbatchvector[2]))
    vectorfile.write("{} \n".format(sti10batchvector[2]))
    vectorfile.write("{} \n".format(chestxraybatchvector[2]))
    vectorfile.write("{} \n".format(pcambatchvector[2]))
    vectorfile.write("{} \n".format(imagenetbatchvector[2]))

    vectorfile.write("homogeneity for chest\n")
    vectorfile.write("{} \n".format(isicbatchvector[2]))
    vectorfile.write("{} \n".format(sti10batchvector[2]))
    vectorfile.write("{} \n".format(pcambatchvector[2]))
    vectorfile.write("{} \n".format(stl10batchvector[2]))
    vectorfile.write("{} \n".format(dtdbatchvector[2]))
    vectorfile.write("{} \n".format(imagenetbatchvector[2]))

    vectorfile.write("homogeneity for pcam \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[2]))
    vectorfile.write("{} \n".format(isicbatchvector[2]))
    vectorfile.write("{} \n".format(sti10batchvector[2]))
    vectorfile.write("{} \n".format(kimiabatchvector[2]))
    vectorfile.write("{} \n".format(stl10batchvector[2]))
    vectorfile.write("{} \n".format(dtdbatchvector[2]))
    vectorfile.write("{} \n".format(imagenetbatchvector[2]))
    
    vectorfile.write("asm for isic \n")
    vectorfile.write("{} \n".format(stl10batchvector[3]))
    vectorfile.write("{} \n".format(dtdbatchvector[3]))
    vectorfile.write("{} \n".format(sti10batchvector[3]))
    vectorfile.write("{} \n".format(chestxraybatchvector[3]))
    vectorfile.write("{} \n".format(pcambatchvector[3]))
    vectorfile.write("{} \n".format(imagenetbatchvector[3]))

    vectorfile.write("asm for chest\n")
    vectorfile.write("{} \n".format(isicbatchvector[3]))
    vectorfile.write("{} \n".format(sti10batchvector[3]))
    vectorfile.write("{} \n".format(pcambatchvector[3]))
    vectorfile.write("{} \n".format(stl10batchvector[3]))
    vectorfile.write("{} \n".format(dtdbatchvector[3]))
    vectorfile.write("{} \n".format(imagenetbatchvector[3]))

    vectorfile.write("asm for pcam \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[3]))
    vectorfile.write("{} \n".format(isicbatchvector[3]))
    vectorfile.write("{} \n".format(sti10batchvector[3]))
    vectorfile.write("{} \n".format(kimiabatchvector[3]))
    vectorfile.write("{} \n".format(stl10batchvector[3]))
    vectorfile.write("{} \n".format(dtdbatchvector[3]))
    vectorfile.write("{} \n".format(imagenetbatchvector[3]))

    vectorfile.write("energy for isic \n")
    vectorfile.write("{} \n".format(stl10batchvector[4]))
    vectorfile.write("{} \n".format(dtdbatchvector[4]))
    vectorfile.write("{} \n".format(sti10batchvector[4]))
    vectorfile.write("{} \n".format(chestxraybatchvector[4]))
    vectorfile.write("{} \n".format(pcambatchvector[4]))
    vectorfile.write("{} \n".format(imagenetbatchvector[4]))

    vectorfile.write("energy for chest\n")
    vectorfile.write("{} \n".format(isicbatchvector[4]))
    vectorfile.write("{} \n".format(sti10batchvector[4]))
    vectorfile.write("{} \n".format(pcambatchvector[4]))
    vectorfile.write("{} \n".format(stl10batchvector[4]))
    vectorfile.write("{} \n".format(dtdbatchvector[4]))
    vectorfile.write("{} \n".format(imagenetbatchvector[4]))

    vectorfile.write("energy for pcam \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[4]))
    vectorfile.write("{} \n".format(isicbatchvector[4]))
    vectorfile.write("{} \n".format(sti10batchvector[4]))
    vectorfile.write("{} \n".format(kimiabatchvector[4]))
    vectorfile.write("{} \n".format(stl10batchvector[4]))
    vectorfile.write("{} \n".format(dtdbatchvector[4]))
    vectorfile.write("{} \n".format(imagenetbatchvector[4]))

    vectorfile.write("correlation for isic \n")
    vectorfile.write("{} \n".format(stl10batchvector[5]))
    vectorfile.write("{} \n".format(dtdbatchvector[5]))
    vectorfile.write("{} \n".format(sti10batchvector[5]))
    vectorfile.write("{} \n".format(chestxraybatchvector[5]))
    vectorfile.write("{} \n".format(pcambatchvector[5]))
    vectorfile.write("{} \n".format(imagenetbatchvector[5]))

    vectorfile.write("correlation for chest\n")
    vectorfile.write("{} \n".format(isicbatchvector[5]))
    vectorfile.write("{} \n".format(sti10batchvector[5]))
    vectorfile.write("{} \n".format(pcambatchvector[5]))
    vectorfile.write("{} \n".format(stl10batchvector[5]))
    vectorfile.write("{} \n".format(dtdbatchvector[5]))
    vectorfile.write("{} \n".format(imagenetbatchvector[5]))

    vectorfile.write("correlation for pcam \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[5]))
    vectorfile.write("{} \n".format(isicbatchvector[5]))
    vectorfile.write("{} \n".format(sti10batchvector[5]))
    vectorfile.write("{} \n".format(kimiabatchvector[5]))
    vectorfile.write("{} \n".format(stl10batchvector[5]))
    vectorfile.write("{} \n".format(dtdbatchvector[5]))
    vectorfile.write("{} \n".format(imagenetbatchvector[5]))

    vectorfile.write("contrast for excel \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[0]))
    vectorfile.write("{} \n".format(isicbatchvector[0]))
    vectorfile.write("{} \n".format(sti10batchvector[0]))
    vectorfile.write("{} \n".format(kimiabatchvector[0]))
    vectorfile.write("{} \n".format(stl10batchvector[0]))
    vectorfile.write("{} \n".format(dtdbatchvector[0]))
    vectorfile.write("{} \n".format(imagenetbatchvector[0]))
    vectorfile.write("{} \n".format(pcambatchvector[0]))

    vectorfile.write("dissimilarity for excel \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[1]))
    vectorfile.write("{} \n".format(isicbatchvector[1]))
    vectorfile.write("{} \n".format(sti10batchvector[1]))
    vectorfile.write("{} \n".format(kimiabatchvector[1]))
    vectorfile.write("{} \n".format(stl10batchvector[1]))
    vectorfile.write("{} \n".format(dtdbatchvector[1]))
    vectorfile.write("{} \n".format(imagenetbatchvector[1]))
    vectorfile.write("{} \n".format(pcambatchvector[1]))

    vectorfile.write("homogeneity for excel \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[2]))
    vectorfile.write("{} \n".format(isicbatchvector[2]))
    vectorfile.write("{} \n".format(sti10batchvector[2]))
    vectorfile.write("{} \n".format(kimiabatchvector[2]))
    vectorfile.write("{} \n".format(stl10batchvector[2]))
    vectorfile.write("{} \n".format(dtdbatchvector[2]))
    vectorfile.write("{} \n".format(imagenetbatchvector[2]))
    vectorfile.write("{} \n".format(pcambatchvector[2]))

    vectorfile.write("asm for excel \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[3]))
    vectorfile.write("{} \n".format(isicbatchvector[3]))
    vectorfile.write("{} \n".format(sti10batchvector[3]))
    vectorfile.write("{} \n".format(kimiabatchvector[3]))
    vectorfile.write("{} \n".format(stl10batchvector[3]))
    vectorfile.write("{} \n".format(dtdbatchvector[3]))
    vectorfile.write("{} \n".format(imagenetbatchvector[3]))
    vectorfile.write("{} \n".format(pcambatchvector[3]))

    vectorfile.write("energy for excel \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[4]))
    vectorfile.write("{} \n".format(isicbatchvector[4]))
    vectorfile.write("{} \n".format(sti10batchvector[4]))
    vectorfile.write("{} \n".format(kimiabatchvector[4]))
    vectorfile.write("{} \n".format(stl10batchvector[4]))
    vectorfile.write("{} \n".format(dtdbatchvector[4]))
    vectorfile.write("{} \n".format(imagenetbatchvector[4]))
    vectorfile.write("{} \n".format(pcambatchvector[4]))

    vectorfile.write("correlation for excel \n")
    vectorfile.write("{} \n".format(chestxraybatchvector[5]))
    vectorfile.write("{} \n".format(isicbatchvector[5]))
    vectorfile.write("{} \n".format(sti10batchvector[5]))
    vectorfile.write("{} \n".format(kimiabatchvector[5]))
    vectorfile.write("{} \n".format(stl10batchvector[5]))
    vectorfile.write("{} \n".format(dtdbatchvector[5]))
    vectorfile.write("{} \n".format(imagenetbatchvector[5]))
    vectorfile.write("{} \n".format(pcambatchvector[5]))

    vectorfile.close()


def normalize3(a, b ,c):
    max = 0
    maxa = np.amax(a)
    maxb = np.amax(b)
    maxc = np.amax(c)

    if maxa > maxb and maxa > maxc:
        max = maxa
    elif maxb > maxa and maxb > maxc:
        max = maxb
    elif maxc > maxa and maxc > maxb:
        max = maxc
    
    a = a/max
    b = b/max
    c = c/max

    return a, b, c

def normalize7(a, b, c, d, e, f, g):
    max = 0
    maxa = np.amax(a)
    maxb = np.amax(b)
    maxc = np.amax(c)
    maxd = np.amax(d)
    maxe = np.amax(e)
    maxf = np.amax(f)
    maxg = np.amax(g)

    maxall = []
    maxall.append(maxa)
    maxall.append(maxb)
    maxall.append(maxc)
    maxall.append(maxd)
    maxall.append(maxe)
    maxall.append(maxf)
    maxall.append(maxg)
    maxall = np.array(maxall)

    maxval = np.amax(maxall)

    a = a/maxval
    b = b/maxval
    c = c/maxval
    d = d/maxval
    e = e/maxval
    f = f/maxval
    g = g/maxval

    return a, b, c, d, e, f, g