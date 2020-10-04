import numpy as np
import keras
from scipy.stats import kurtosis, skew, entropy as scipy_entropy

# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# cifar10 = keras.datasets.cifar10
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
#
# img = train_images[0]
# img2 = train_images[1]
#
# dataset = train_images[:2]

#%%
# Shannon entropy
def meta_entropy(image, base=2):
    _, counts = np.unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

# print(entropy(img, base=len(img.shape)), entropy(img2, base=len(img2.shape)))

## Shannon entropy
# marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
# marg = list(filter(lambda p: p > 0, np.ravel(marg)))
# shannon_entropy = -np.sum(np.multiply(marg, np.log2(marg)))
# print(shannon_entropy)

#%% Skewness
def meta_skew(image):
    skewness = skew(image, axis=None)
    return skewness

# print(meta_skew(img), meta_skew(img2))
#%% Kurtosis
def meta_kurtosis(image):
    kurt = kurtosis(image, axis=None)
    return kurt

# print(meta_kurtosis(img), meta_kurtosis(img2))
#%% Median
def meta_median(image):
    median = np.median(image)
    return median

# print(meta_median(img), meta_median(img2))
#%% Std
def meta_std(image):
    std = np.std(image)
    return std

# print(meta_std(img), meta_std(img2))
#%% Mean
def meta_mean(image):
    mean = np.mean(image)
    return mean

# print(meta_mean(img), meta_mean(img2))
#%% Sparsity
def meta_sparsity(image):
    sparsity = np.count_nonzero(image)/np.prod(image.shape)
    return sparsity

# print(meta_sparsity(img), meta_sparsity(img2))
#%% XY-axis
def meta_xy_axis(image):
    xy_axis = (image.shape[0]+image.shape[1])/2
    return xy_axis

# print(meta_xy_axis(img), meta_xy_axis(img2))
#%% Color image section

# RGB calculations

def meta_rgb(image):
    red = []
    green = []
    blue = []

    for column in range(len(image)):
        for pixel in range(len(image[column])):
            red.append(image[column][pixel][0])
            green.append(image[column][pixel][1])
            blue.append(image[column][pixel][2])

    mean_red = np.mean(red)
    mean_green = np.mean(green)
    mean_blue = np.mean(blue)

# Z-axis
def meta_z_axis(image):
    z_axis = image.shape[2]
    return z_axis

# print(z_axis)
#%% Image comparisons within a dataset
# Mutual information

# Mean squared error

# Structural similarity measure

# Correlation coefficient
# cor = signal.correlate2d(img, img2)
# print(cor)

#%% Test section

# cifar10 = keras.datasets.cifar10
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# image = train_images[0]
# column = train_images[0][0]
# print(column)

# for i in range(len(column)):
#     column[i] = np.array([(0.2989 * column[i][0] + 0.5870 * column[i][1] + 0.1140 * column[i][2])])
# print(column)
# column = column[:, :1]
# print(column)

# for x in range(1):                  # image[x] is the column
#     print(image[x])
#     for i in range(len(image[x])):
#         image[x][i] = np.array([(0.2989 * image[x][i][0] + 0.5870 * image[x][i][1] + 0.1140 * image[x][i][2])])
#     print(image[x])
#     image[x] = image[x][:, :1]
#     print(image[x])

# print(image)