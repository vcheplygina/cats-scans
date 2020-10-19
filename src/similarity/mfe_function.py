import keras
import numpy as np
import os
from src.io.converter_numpy import get_train_images
from src.similarity.meta_features import meta_entropy, meta_skew, meta_kurtosis, meta_median, meta_std, meta_mean, meta_sparsity, meta_xy_axis

#%% Function

def feature_extraction(datasets, mfe_path, mfe_subset):
    """Calculates the similarity vector between datasets based on meta-features"""
    # Importing datasets and defining subsets

    data_vectors = []

    for dataset in datasets:
        if dataset == 'mnist':
            mnist = keras.datasets.mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        elif dataset == 'fashion_mnist':
            fashion_mnist = keras.datasets.fashion_mnist
            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        elif dataset == 'cifar10':
            cifar10 = keras.datasets.cifar10
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        elif dataset == 'cifar100':
            cifar100 = keras.datasets.cifar100
            (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
        elif dataset == 'ISIC2017':
            train_images = get_train_images(dataset='ISIC2017', converter_path=mfe_path, converter_subset=mfe_subset)
            train_labels = 7
        elif dataset == 'ISIC2018':
            train_images = get_train_images(dataset='ISIC2018', converter_path=mfe_path, converter_subset=mfe_subset)
            train_labels = 7
        elif dataset == 'chest_xray':
            train_images = get_train_images(dataset='chest_xray', converter_path=mfe_path, converter_subset=mfe_subset)
            train_labels = len(next(os.walk(mfe_path + '/chest_xray/train/'))[1])
        elif dataset == 'stl-10':
            train_images = get_train_images(dataset='stl-10', converter_path=mfe_path, converter_subset=mfe_subset)
            train_labels = len(next(os.walk(mfe_path + '/stl_10/'))[1])
        elif dataset == 'dtd':
            train_images = get_train_images(dataset='dtd', converter_path=mfe_path, converter_subset=mfe_subset)
            train_labels = len(next(os.walk(mfe_path + '/dtd/'))[1])
        elif dataset == 'pcam':
            train_images = get_train_images(dataset='pcam', converter_path=mfe_path, converter_subset=mfe_subset)
            train_labels = len(next(os.walk(mfe_path + '/pcam/pcam_subset/'))[1])
        else:
            print('Dataset not available')
            break

        if dataset == 'mnist' or 'fashion_mnist' or 'cifar10' or 'cifar100':
            data = train_images[:mfe_subset]
        else:
            data = train_images

        # General meta-features

        if dataset == 'chest_xray' or dataset == 'dtd' or dataset == 'pcam':
            image_count = data.shape[0]
            image_size = data[0].shape[0] * data[0].shape[1]

        else:
            image_count = data.shape[0]
            image_size = data.shape[1] * data.shape[2]

        label_count = train_labels

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
        meta_vector = [image_count, image_size, label_count, entropy_mean, entropy_std, skewness_mean, skewness_std, kurtosis_mean, kurtosis_std, median_mean,
                        median_std, std_mean, std_std, mean_mean, mean_std, sparsity_mean, sparsity_std, xy_axis_mean, xy_axis_std]

        # Zero mean and unit variance

        meta_vector_mean = np.mean(meta_vector)
        meta_vector_std = np.std(meta_vector)

        for meta_feature in range(len(meta_vector)):
            meta_vector[meta_feature] = np.round(((meta_vector[meta_feature] - meta_vector_mean) / meta_vector_std), decimals=2)

        # Combine dataset vectors

        data_vectors.append(meta_vector)

        print(dataset + ':', 'image count = ' + str(image_count) + ',', 'image size = ' + str(image_size) + ',', 'label count = ' + str(label_count))

    # Similarity matrix calculation

    sim_mat = np.zeros((len(data_vectors), len(data_vectors)))

    for row in range(len(data_vectors)):
        for column in range(len(data_vectors)):
            euc_dst = np.linalg.norm(np.asarray(data_vectors[row]) - np.asarray(data_vectors[column]))
            sim_mat[row][column] = np.around(euc_dst, decimals=2)

    return sim_mat

