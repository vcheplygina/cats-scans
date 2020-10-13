import numpy as np
from scipy.stats import kurtosis, skew, entropy as scipy_entropy

#%% Meta-features

# Shannon entropy
def meta_entropy(image, base=2):
    _, counts = np.unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

# Skewness
def meta_skew(image):
    skewness = skew(image, axis=None)
    return skewness

# Kurtosis
def meta_kurtosis(image):
    kurt = kurtosis(image, axis=None)
    return kurt

# Median
def meta_median(image):
    median = np.median(image)
    return median

# Standard deviation
def meta_std(image):
    std = np.std(image)
    return std

# Mean
def meta_mean(image):
    mean = np.mean(image)
    return mean

# Sparsity
def meta_sparsity(image):
    sparsity = np.count_nonzero(image)/np.prod(image.shape)
    return sparsity

# XY-axis
def meta_xy_axis(image):
    xy_axis = (image.shape[0]+image.shape[1])/2
    return xy_axis