from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

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

