from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

#%% Get local images and convert them to numpy arrays

def get_train_images(dataset = 'ISIC2018', local_subset = 5):
    """Retrieve files from local depository and convert them to numpy arrays"""
    begin_path = 'C:/Users/20169385/PycharmProjects/cats-scans/local_data/datasets'
    if dataset == 'ISIC2017':
        path_to_dataset = '/ISIC2017/ISIC2017_Task3_Training_Input/'
    elif dataset == 'ISIC2018':
        path_to_dataset = '/ISIC2018/ISIC2018_Task3_Training_Input/'
    elif dataset == 'chest_xray':
        path_to_dataset = '/chest_xray/train/all/'
    elif dataset == 'stl-10':
        path_to_dataset = '/stl_10/all/'
    elif dataset == 'dtd':
        path_to_dataset = '/dtd/all/'
    else:
        return None

    dataset_filenames = [f for f in listdir(begin_path + path_to_dataset) if
                         isfile(join(begin_path + path_to_dataset, f))][:local_subset]

    train_images = []

    for image_name in dataset_filenames:
        image = Image.open(begin_path + path_to_dataset + image_name)
        image = np.asarray(image)
        train_images.append(image)

    train_images = np.array(train_images)

    return train_images