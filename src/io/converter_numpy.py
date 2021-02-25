# Import packages
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2

def get_train_images(dataset, converter_path, converter_subset):
    """Retrieves files from a local depository and converts them to numpy array.
     The converter_subset argument is defined in the similarity_experiment file and
     indicates the size of the subset and thus the amount of images to be retrieved from the directory."""

    begin_path = converter_path

    # The dataset images are stored in different folders. Below the right folder path is specified according to the dataset input argument

    if dataset == 'ISIC2018':
        path_to_dataset = '/ISIC2018/'
    elif dataset == 'chest_xray':
        path_to_dataset = '/chest_xray/all/'
    elif dataset == 'stl-10':
        path_to_dataset = '/stl_10/'
    elif dataset == 'dtd':
        path_to_dataset = '/dtd/'
    elif dataset == 'pcam':
        path_to_dataset = '/pcam/pcam_subset/'
    else:
        return None

    # Loop over every subdirectory to get all the images and make empty lists

    sub_dirs = next(os.walk(begin_path + path_to_dataset))[1]
    train_images = []
    names_count = []

    # Retrieve all images and resize them to 300 by 300

    for index in range(len(sub_dirs)):
        if converter_subset == 'None':
            converter_subset = 15000                # If size of subset is defined as None, a maximum of 15000 is taken to avoid memory error
        else:
            converter_subset = converter_subset

        all_names = listdir(begin_path + path_to_dataset + sub_dirs[index])
        for i in range(len(all_names)):
            names_count.append(all_names[i])

        if len(names_count) < converter_subset:
            dataset_filenames = [f for f in listdir(begin_path + path_to_dataset + sub_dirs[index]) if
                                 isfile(join(begin_path + path_to_dataset + sub_dirs[index], f))]
        else:
            dataset_filenames = [f for f in listdir(begin_path + path_to_dataset + sub_dirs[index]) if
                                 isfile(join(begin_path + path_to_dataset + sub_dirs[index], f))][:(round(converter_subset/len(sub_dirs)))]
        for image_name in dataset_filenames:
            image = cv2.imread(begin_path + path_to_dataset + sub_dirs[index] + '/' + image_name)
            image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
            train_images.append(image)

    train_images = np.array(train_images)

    return train_images
