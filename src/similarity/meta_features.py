import numpy as np
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
import cv2

#%% Meta-features

# Shannon entropy
def meta_entropy(image, base=2):
    image = cv2.calcHist(image, [0], None, [256], [0, 256])
    _, counts = np.unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

# Skewness
def meta_skew(image):
    image = cv2.calcHist(image, [0], None, [256], [0, 256])
    skewness = skew(image, axis=None)
    return skewness

# Kurtosis
def meta_kurtosis(image):
    image = cv2.calcHist(image, [0], None, [256], [0, 256])
    kurt = kurtosis(image, axis=None)
    return kurt

# Standard deviation
def meta_std(image):
    image = cv2.calcHist(image, [0], None, [256], [0, 256])
    std = np.std(image)
    return std

# Mean
def meta_mean(image):
    image = cv2.calcHist(image, [0], None, [256], [0, 256])
    mean = np.mean(image)
    return mean

# Sparsity
def meta_sparsity(image):
    image = cv2.calcHist(image, [0], None, [256], [0, 256])
    sparsity = np.count_nonzero(image)/np.prod(image.shape)
    return sparsity
