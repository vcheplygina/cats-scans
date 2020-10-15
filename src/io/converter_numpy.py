from PIL import Image
import numpy as np
import os
from os import listdir
from os.path import isfile, join

#%% Get local images and convert them to numpy arrays

def get_train_images(dataset = 'ISIC2018', local_subset = 50):
    """Retrieve files from local depository and convert them to numpy arrays"""

    begin_path = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets'
    if dataset == 'ISIC2017':
        path_to_dataset = '/ISIC2017/'
    elif dataset == 'ISIC2018':
        path_to_dataset = '/ISIC2018/'
    elif dataset == 'chest_xray':
        path_to_dataset = '/chest_xray/train/'
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

    for index in range(len(sub_dirs)):
        dataset_filenames = [f for f in listdir(begin_path + path_to_dataset + sub_dirs[index]) if
                             isfile(join(begin_path + path_to_dataset + sub_dirs[index], f))][:(round(local_subset/len(sub_dirs)))]
        for image_name in dataset_filenames:
            image = Image.open(begin_path + path_to_dataset + sub_dirs[index] + '/' + image_name)
            image = np.asarray(image)
            train_images.append(image)

    train_images = np.array(train_images)

    return train_images

#%%

subset = 50
local_subset = 50
begin_path = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets'
path_to_dataset = '/dtd/'

sub_dirs = next(os.walk(begin_path + path_to_dataset))[1]
# print(sub_dirs)

train_images = []

for index in range(len(sub_dirs)):
    dataset_filenames = [f for f in listdir(begin_path + path_to_dataset + sub_dirs[index]) if
                         isfile(join(begin_path + path_to_dataset + sub_dirs[index], f))][
                        :(round(local_subset / len(sub_dirs)))]
    print(dataset_filenames)

