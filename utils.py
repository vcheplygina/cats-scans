import numpy as np
import keras
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from scipy import signal

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

img = train_images[0]
img2 = train_images[2]

#%%
# Shannon entropy
def entropy(image, base=2):
    _, counts = np.unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

# # Shannon entropy
# marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
# marg = list(filter(lambda p: p > 0, np.ravel(marg)))
# shannon_entropy = -np.sum(np.multiply(marg, np.log2(marg)))
# print(shannon_entropy)

# Skewness
def skew(image):
    skewness = skew(image, axis=None)
    return skewness

# Kurtosis
def kurtosis(image):
    kurt = kurtosis(image, axis=None)
    return kurt

# Median
def median(image):
    median = np.median(image)
    return median

# Std
def std(image):
    std = np.std(image)
    return std

# Mean
def mean(image):
    mean = np.mean(image)
    return mean

# Sparsity
def sparsity(image):
    sparsity = np.count_nonzero(image)/np.prod(image.shape)
    return sparsity

# XY-axis
def xy_axis(image):
    xy_axis = (image.shape[0]+image.shape[1])/2
    return xy_axis

#%% Color image section

# RGB calculations

# Z-axis (only for color images)
# z_axis = img.shape[2]
# print(z_axis)

#%% Image comparisons within a dataset
# Mutual information

# Mean squared error

# Structural similarity measure

# Correlation coefficient
# cor = signal.correlate2d(img, img2)
# print(cor)

#%% Test

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
image = train_images[0]
# column = train_images[0][0]
# print(column)

# for i in range(len(column)):
#     column[i] = np.array([(0.2989 * column[i][0] + 0.5870 * column[i][1] + 0.1140 * column[i][2])])
# print(column)
# column = column[:, :1]
# print(column)

for x in range(1): # image[x] is the column
    print(image[x])
    for i in range(len(image[x])):
        image[x][i] = np.array([(0.2989 * image[x][i][0] + 0.5870 * image[x][i][1] + 0.1140 * image[x][i][2])])
    print(image[x].shape)
    image[x] = image[x][:, :1]
    print(image[x])

# print(image)