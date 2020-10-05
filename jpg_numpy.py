from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
#%%

def get_train_images(dataset = 'ISIC2018'):
    """Retrieve files from local depository and convert them to numpy arrays"""
    if dataset == 'ISIC2018':
        path_to_dataset = 'datasets/ISIC2018/ISIC2018_Task3_Training_Input/'
    elif dataset == 'chest_xray':
        path_to_dataset = 'datasets/chest_xray/train/NORMAL/'
    else:
        return None

    dataset_filenames = [f for f in listdir(path_to_dataset) if
                         isfile(join(path_to_dataset, f))][:5]  # Take only 5 images for fast computation times

    train_images = []

    for image_name in dataset_filenames:
        image = Image.open(path_to_dataset + image_name)
        image = np.asarray(image)
        train_images.append(image)

    train_images = np.array(train_images)

    return train_images

get_train_images(dataset = 'ISIC2018')

#%%
im = Image.open('datasets/ISIC2018/ISIC2018_Task3_Training_Input/ISIC_0024306.jpg')
im2 = Image.open('datasets/chest_xray/train/NORMAL/IM-0115-0001.jpeg')
#
# # im = Image.open('datasets/chest_xray/train/NORMAL/IM-0115-0001.jpeg')
# # im2 = Image.open('datasets/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg')
#
im = np.asarray(im)
im2 = np.asarray(im2)

print(im2, im2.shape)

# images = [im, im2]
#
# print(np.array(images).shape)
#%%

# from os import listdir
# from os.path import isfile, join
# dataset_filenames = [f for f in listdir('datasets/ISIC2018/ISIC2018_Task3_Training_Input') if isfile(join('datasets/ISIC2018/ISIC2018_Task3_Training_Input', f))][:5]
#
# print(dataset_filenames)

#%%
# from PIL import Image
# import numpy as np
# import keras
# from os import listdir
# from os.path import isfile, join
#
# train_images = []
#
# for image_name in dataset_filenames:
#     image = Image.open('datasets/ISIC2018/ISIC2018_Task3_Training_Input/'+ image_name)
#     image = np.asarray(image)
#     train_images.append(image)
#
# train_images = np.array(train_images)
#
# print(train_images)

#%%

# def jpg_image_to_array(image_path):
#   """
#   Loads JPEG image into 3D Numpy array of shape
#   (width, height, channels)
#   """
#   with Image.open(image_path) as image:
#     im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
#     im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
#   return im_arr
#
# jpg_image_to_array('ISIC_0000000')

#%% Test

# import keras

# cifar10 = keras.datasets.cifar10
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# train_images = train_images[:2]
#
# print(len(train_labels))

