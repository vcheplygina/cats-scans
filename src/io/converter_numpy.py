import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2

#%% Get local images and convert them to numpy arrays

def get_train_images(dataset, converter_path, converter_subset):
    """Retrieve files from local depository and convert them to numpy arrays"""

    begin_path = converter_path
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

    sub_dirs = next(os.walk(begin_path + path_to_dataset))[1]
    train_images = []
    names_count = []

    for index in range(len(sub_dirs)):
        if converter_subset == 'None':
            converter_subset = 25000        # to avoid memory error
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
            image = cv2.imread(begin_path + path_to_dataset + sub_dirs[index] + '/' + image_name, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
            train_images.append(image)

    train_images = np.array(train_images)

    return train_images

