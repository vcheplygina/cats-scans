import keras
import numpy as np
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from PIL import Image
from os import listdir
from os.path import isfile, join
# from jpg_numpy import get_train_images
# from meta_features import meta_entropy, meta_skew, meta_kurtosis, meta_median, meta_std, meta_mean, meta_sparsity, meta_xy_axis

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

#%% Get local jpg images and convert them to numpy arrays

def get_train_images(dataset = 'ISIC2018', subset = 5):
    """Retrieve files from local depository and convert them to numpy arrays"""
    if dataset == 'ISIC2017':
        path_to_dataset = 'datasets/ISIC2017/ISIC2017_Task3_Training_Input/'
    elif dataset == 'ISIC2018':
        path_to_dataset = 'datasets/ISIC2018/ISIC2018_Task3_Training_Input/'
    elif dataset == 'chest_xray':
        path_to_dataset = 'datasets/chest_xray/train/NORMALJPG/'
    else:
        return None

    dataset_filenames = [f for f in listdir(path_to_dataset) if
                         isfile(join(path_to_dataset, f))][:subset]  # Take only 5 images for fast computation times

    train_images = []

    for image_name in dataset_filenames:
        image = Image.open(path_to_dataset + image_name)
        image = np.asarray(image)
        train_images.append(image)

    train_images = np.array(train_images)

    return train_images

#%% Function

def feature_extraction(datasets, subset = 20):
    """Calculates the similarity vector based on meta-features between datasets"""
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
            (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
            data = train_images[:subset]
        elif data == 'ISIC2017':
            train_images = get_train_images(dataset='ISIC2017', subset=subset)
            train_labels = 2
            data = train_images[:subset]
        elif data == 'ISIC2018':
            train_images = get_train_images(dataset='ISIC2018', subset=subset)
            train_labels = 2
            data = train_images[:subset]
        elif data == 'chest_xray':
            train_images = get_train_images(dataset='chest_xray', subset=subset)
            train_labels = 2
            data = train_images[:subset]
        else:
            return None

        # General meta-features

        image_count = data.shape[0]
        image_size = data.shape[1] * data.shape[2]
        label_count = len(np.unique(train_labels))-1

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
            data_entropy.append(meta_entropy(image, base=len(image.shape)))
            data_skewness.append(meta_skew(image))
            data_kurtosis.append(meta_kurtosis(image))
            data_median.append(meta_median(image))
            data_std.append(meta_std(image))
            data_mean.append(meta_mean(image))
            data_sparsity.append(meta_sparsity(image))
            data_xy_axis.append(meta_xy_axis(image))

        entropy_mean = round(np.mean(data_entropy))
        entropy_std = round(np.std(data_entropy))

        skewness_mean = round(np.mean(data_skewness))
        skewness_std = round(np.std(data_skewness))

        kurtosis_mean = round(np.mean(data_kurtosis))
        kurtosis_std = round(np.std(data_kurtosis))

        median_mean = round(np.mean(data_median))
        median_std = round(np.std(data_median))

        std_mean = round(np.mean(data_std))
        std_std = round(np.std(data_std))

        mean_mean = round(np.mean(data_mean))
        mean_std = round(np.std(data_mean))

        sparsity_mean = round(np.mean(data_sparsity))
        sparsity_std = round(np.std(data_sparsity))

        xy_axis_mean = round(np.mean(data_xy_axis))
        xy_axis_std = round(np.std(data_xy_axis))

        # Combine meta-features
        all_features = [image_count, image_size, label_count, entropy_mean, entropy_std, skewness_mean, skewness_std, kurtosis_mean, kurtosis_std, median_mean,
                        median_std, std_mean, std_std, mean_mean, mean_std, sparsity_mean, sparsity_std, xy_axis_mean, xy_axis_std]

        meta_features = []

        for index in range(len(all_features)):
            meta_features.append(all_features[index])

        # Combine dataset vectors

        data_vectors.append(meta_features)

    # Similarity matrix calculation

    sim_mat = np.zeros((len(data_vectors), len(data_vectors)))

    for row in range(len(data_vectors)):
        for column in range(len(data_vectors)):
            euc_dst = np.linalg.norm(np.asarray(data_vectors[row]) - np.asarray(data_vectors[column]))
            sim_mat[row][column] = np.around(euc_dst, decimals=2)

    return sim_mat

feature_extraction(datasets=['ISIC2017', 'ISIC2018'], subset=10)

#%% Test