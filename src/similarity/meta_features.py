# Import packages
import numpy as np
from scipy.stats import kurtosis, skew, entropy, median_absolute_deviation

# Meta-feature function

# Entropy
def meta_entropy(image, base=2):
    """Measure for the randomness of the distribution"""
    im_entropy = entropy(image)
    return im_entropy

# Skewness
def meta_skew(image):
    """Degree of asymmetry"""
    skewness = skew(image, axis=None)
    return skewness

# Kurtosis
def meta_kurtosis(image):
    """Relative peakedness or flatness of distribution"""
    kurt = kurtosis(image, axis=None)
    return kurt

# Median absolute deviation
def meta_mad(image):
    """Absolute dispersion of the frequency of the values"""
    mad = median_absolute_deviation(image)
    return mad

# Mean
def meta_mean(image):
    """Central tendency of the histogram"""
    mean = np.mean(image)
    return mean

# Variance
def meta_variance(image):
    """Describes how far the values lie from the mean"""
    sparsity = np.var(image)
    return sparsity
