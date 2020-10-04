import keras
import numpy as np
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
# from utils import meta_entropy, meta_skew, meta_kurtosis, meta_median, meta_std, meta_mean, meta_sparsity, meta_xy_axis
#%% Utils

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

#%% Function

def feature_extraction(datasets, subset = 20):
    """calculates the similarity vector based on meta-features between datasets"""
    # Importing datasets and defining subsets

    data_vectors = []

    for data in datasets:
        if data == 'mnist':
            mnist = keras.datasets.mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            data = train_images[:subset]
        elif data == 'fashion_mnist':
            fashion_mnist = keras.datasets.fashion_mnist
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
            data = train_images[:subset]
        elif data == 'cifar10':
            cifar10 = keras.datasets.cifar10
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
            data = train_images[:subset]
        elif data == 'cifar100':
            cifar100 = keras.datasets.cifar100
            (train_images2, train_labels2), (test_images2, test_labels2) = cifar100.load_data()
            data = train_images[:subset]
        else:
            return None

        # General meta-features

        image_count = data.shape[0]
        image_size = data.shape[1] * data.shape[2]
        label_count = len(np.unique((train_labels)))-1

        # Statistical meta-features

        data_entropy = []
        data_skewness = []
        data_kurtosis = []
        data_median = []
        data_std = []
        data_mean = []
        data_sparsity = []
        data_xy_axis = []

        for image in data:
            data_entropy.append(round(meta_entropy(image, base=len(image.shape)), 2))
            data_skewness.append(round(meta_skew(image), 2))
            data_kurtosis.append(round(meta_kurtosis(image), 2))
            data_median.append(round(meta_median(image), 2))
            data_std.append(round(meta_std(image), 2))
            data_mean.append(round(meta_mean(image), 2))
            data_sparsity.append(round(meta_sparsity(image), 2))
            data_xy_axis.append(round(meta_xy_axis(image), 2))

        entropy_mean = np.mean(data_entropy)
        entropy_std = np.std(data_entropy)

        skewness_mean = np.mean(data_skewness)
        skewness_std = np.std(data_skewness)

        kurtosis_mean = np.mean(data_kurtosis)
        kurtosis_std = np.std(data_kurtosis)

        median_mean = np.mean(data_median)
        median_std = np.std(data_median)

        std_mean = np.mean(data_std)
        std_std = np.std(data_std)

        mean_mean = np.mean(data_mean)
        mean_std = np.std(data_mean)

        sparsity_mean = np.mean(data_sparsity)
        sparsity_std = np.std(data_sparsity)

        xy_axis_mean = np.mean(data_xy_axis)
        xy_axis_std = np.std(data_xy_axis)

        # Combine meta-features
        all_features = [image_count, image_size, label_count, entropy_mean, entropy_std, skewness_mean, skewness_std, kurtosis_mean, kurtosis_std, median_mean,
                        median_std, std_mean, std_std, mean_mean, mean_std, sparsity_mean, sparsity_std, xy_axis_mean, xy_axis_std]

        meta_features = []

        for index in range(len(all_features)):
            meta_features.append(all_features[index])

        # Combine dataset vectors

        data_vectors.append(meta_features)

    return data_vectors

feature_extraction(datasets=['mnist', 'cifar10'], subset=10)

#%% Test