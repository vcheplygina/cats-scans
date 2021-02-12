# Import packages
import numpy as np
import os
from src.io.converter_numpy import get_train_images
from src.similarity.meta_features import meta_entropy, meta_skew, meta_kurtosis, meta_mad, meta_mean, meta_variance
import cv2

def feature_extraction(datasets, mfe_path, mfe_subset, color_channel):
    """Calculates the similarity between datasets based on statistical meta-features.
    Takes five possible datasets as input arguments and returns a matrix with the corresponding similarity measures."""

    # Importing datasets and defining subsets

    data_vectors = []

    for dataset in datasets:
        if dataset == 'ISIC2018':
            train_images = get_train_images(dataset='ISIC2018', converter_path=mfe_path, converter_subset=mfe_subset)
            num_train_labels = 7
        elif dataset == 'chest_xray':
            train_images = get_train_images(dataset='chest_xray', converter_path=mfe_path, converter_subset=mfe_subset)
            num_train_labels = len(next(os.walk(mfe_path + '/chest_xray/all/'))[1])
        elif dataset == 'stl-10':
            train_images = get_train_images(dataset='stl-10', converter_path=mfe_path, converter_subset=mfe_subset)
            num_train_labels = len(next(os.walk(mfe_path + '/stl_10/'))[1])
        elif dataset == 'dtd':
            train_images = get_train_images(dataset='dtd', converter_path=mfe_path, converter_subset=mfe_subset)
            num_train_labels = len(next(os.walk(mfe_path + '/dtd/'))[1])
        elif dataset == 'pcam':
            train_images = get_train_images(dataset='pcam', converter_path=mfe_path, converter_subset=mfe_subset)
            num_train_labels = len(next(os.walk(mfe_path + '/pcam/pcam_subset/'))[1])
        else:
            print('Dataset not available')
            break

        data = train_images

        # General meta-features

        image_count = data.shape[0]
        label_count = num_train_labels

        # Make empty lists for statistical meta-features

        data_entropy = []
        data_skewness = []
        data_kurtosis = []
        data_mad = []
        data_mean = []
        data_variance = []

        # Convert to color channel histograms

        for image in data:
            if color_channel == 'blue':
                image = cv2.calcHist([image], [0], None, [256], [0, 256])
            elif color_channel == 'green':
                image = cv2.calcHist([image], [1], None, [256], [0, 256])
            elif color_channel == 'red':
                image = cv2.calcHist([image], [2], None, [256], [0, 256])
            data_entropy.append(meta_entropy(image, base=len(image.shape)))
            data_skewness.append(meta_skew(image))
            data_kurtosis.append(meta_kurtosis(image))
            data_mad.append(meta_mad(image))
            data_mean.append(meta_mean(image))
            data_variance.append(meta_variance(image))

        # Take the mean and std of all the statistical values within each dataset

        entropy_mean = np.mean(data_entropy)
        entropy_std = np.std(data_entropy)

        skewness_mean = np.mean(data_skewness)
        skewness_std = np.std(data_skewness)

        kurtosis_mean = np.mean(data_kurtosis)
        kurtosis_std = np.std(data_kurtosis)

        mad_mean = np.mean(data_mad)
        mad_std = np.std(data_mad)

        mean_mean = np.mean(data_mean)
        mean_std = np.std(data_mean)

        variance_mean = np.mean(data_variance)
        variance_std = np.std(data_variance)

        # Combine statistical meta-features into one vector

        meta_vector = [image_count, label_count, entropy_mean, entropy_std, skewness_mean, skewness_std, kurtosis_mean, kurtosis_std,
                       mad_mean, mad_std, mean_mean, mean_std, variance_mean, variance_std]

        # Convert vectors to zero mean and unit variance

        meta_vector_mean = np.mean(meta_vector)
        meta_vector_std = np.std(meta_vector)

        for meta_feature in range(len(meta_vector)):
            meta_vector[meta_feature] = ((meta_vector[meta_feature] - meta_vector_mean) / meta_vector_std)

        # Combine all dataset vectors

        data_vectors.append(meta_vector)

    # Calculate the euclidean distances between the vectors and store them in the similarity matrix

    sim_mat = np.zeros((len(data_vectors), len(data_vectors)))

    for row in range(len(data_vectors)):
        for column in range(len(data_vectors)):
            euc_dst = np.linalg.norm(np.asarray(data_vectors[row]) - np.asarray(data_vectors[column]))
            sim_mat[row][column] = np.around(euc_dst, decimals=2)

    return sim_mat
