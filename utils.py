import numpy as np
import keras
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
import skimage.measure as sm

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

img = train_images[0]
#%%

# Shannon entropy
marg = np.histogramdd(np.ravel(img), bins = 256)[0]/img.size
marg = list(filter(lambda p: p > 0, np.ravel(marg)))
shannon_entropy = -np.sum(np.multiply(marg, np.log2(marg)))
print(shannon_entropy)

# Skewness
skewness = skew(img, axis=None)
print(skewness)

# Kurtosis
kurtosis = kurtosis(img, axis=None)
print(kurtosis)

# Median
median = np.median(img)
print(median)

# Std
std = np.std(img)
print(std)

# Mean
mean = np.mean(img)
print(mean)

# Mutual information (between images within dataset)

# Correlation coefficient (between images within dataset)

# Sparsity
sparsity = np.count_nonzero(img)/np.prod(img.shape)
print(sparsity)

# XY-axis
xy_axis = (img.shape[0]+img.shape[1])/2
print(xy_axis)

# Z-axis (only for color images)
# z_axis = img.shape[2]
# print(z_axis)

