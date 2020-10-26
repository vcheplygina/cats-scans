import numpy as np
from scipy.stats import kurtosis, skew, entropy, median_absolute_deviation

#%% Meta-features

# Shannon entropy
def meta_entropy(image, base=2):
    im_entropy = entropy(image)
    return im_entropy

# Skewness
def meta_skew(image):
    skewness = skew(image, axis=None)
    return skewness

# Kurtosis
def meta_kurtosis(image):
    kurt = kurtosis(image, axis=None)
    return kurt

# Median absolute deviation
def meta_mad(image):
    mad = median_absolute_deviation(image)
    return mad

# Mean
def meta_mean(image):
    mean = np.mean(image)
    return mean

# Variance
def meta_variance(image):
    sparsity = np.var(image)
    return sparsity
