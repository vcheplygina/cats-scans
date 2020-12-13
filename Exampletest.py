from Utils import npytoarray, preprocesser, batchgenerator, visualizer, computeglcm, createvector, compareeucdistance, comparecosine, normalize, bintoarray
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from skimage import io, color
from skimage.filters import gabor, median
from skimage.feature import greycomatrix, greycoprops
import cv2
import os

#Generate batches with indexes, size is set to 512, indexes below 8060 because sti-10 contains 8060 images 
x = batchgenerator(512, 8060, 0)
y = batchgenerator(512, 8060, 0)


#Generate image arrays out of datasets, npytoarray is used for Sti-10, bintoarray for Stl-10
#Make sure one array is commented and one arrayb is commented

array = npytoarray(data='sti10', nums=x)
#array = bintoarray(nums=x, path_to_data='unlabeled_X.bin')
arrayb = npytoarray(data='sti10', nums=y)
#arrayb = bintoarray(nums=y, path_to_data='unlabeled_X.bin')


#Conversion to greyscale and uint8 if needed
grey, filtered = preprocesser(array)
greyb, filteredb = preprocesser(arrayb)


#Compute grey level co-occurence matrix
contrast, dissimilarity, homogeneity, asm, energy, correlation = computeglcm(original_imgs=array, original_grey=grey, grey_filtered=filtered, d=[1], a=[0], levels=256)
contrastb, dissimilarityb, homogeneityb, asmb, energyb, correlationb = computeglcm(original_imgs=arrayb, original_grey=greyb, grey_filtered=filteredb, d=[1], a=[0], levels=256)


#Uncomment to visualize images of 'array', code pauses until new window is closed
#visualizer(nrows=3, ncols=4, nums=x, original_imgs=array, original_grey_imgs=grey, filtered_imgs=filtered)


#Normalization of all features
ncontrast, ncontrastb = normalize(contrast, contrastb)
ndissimilarity, ndissimilarityb = normalize(dissimilarity, dissimilarityb)
nhomogeneity, nhomogeneityb = normalize(homogeneity, homogeneityb)
nasm, nasmb = normalize(asm, asmb)
nenergy, nenergyb = normalize(energy, energyb)
ncorrelation, ncorrelationb = normalize(correlation, correlationb)


#Creation of .txt file with all vector statistics and batchvectors
batchvectora = createvector(name = 'sti10a', nums=x, contrast=ncontrast, dissimilarity=ndissimilarity, homogeneity=nhomogeneity, asm=nasm, energy=nenergy, correlation=ncorrelation)
batchvectorb = createvector(name = 'sti10b', nums=y, contrast=ncontrastb, dissimilarity=ndissimilarityb, homogeneity=nhomogeneityb, asm=nasmb, energy=nenergyb, correlation=ncorrelationb)


#Print comparisons
compareeucdistance(batchvectora, batchvectorb)
comparecosine(batchvectora, batchvectorb)