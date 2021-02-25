import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from .data_paths import get_path
from sklearn import preprocessing
import cv2
from numpy.random import seed
import tensorflow as tf

# set seeds for reproducibility
seed(1)
tf.random.set_seed(2)


def import_ISIC(img_dir, label_dir):
    """
    :param img_dir: directory where images are stored
    :param label_dir: directory where labels are stored
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .jpg
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

    # import labels and set image id as index column
    labels = pd.read_csv(label_dir)
    labels = labels.set_index("image")

    tables = []  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=['path'])  # add img path to dataframe
        img_id = img_path[-16:-4]  # get image id from image path
        extracted_label = labels.loc[img_id]  # extract label from label csv
        if extracted_label[0] == 1:
            extracted_label = 'MEL'
        elif extracted_label[1] == 1:
            extracted_label = 'NV'
        elif extracted_label[2] == 1:
            extracted_label = 'BCC'
        elif extracted_label[3] == 1:
            extracted_label = 'AKIEC'
        elif extracted_label[4] == 1:
            extracted_label = 'BKL'
        elif extracted_label[5] == 1:
            extracted_label = 'DF'
        elif extracted_label[6] == 1:
            extracted_label = 'VASC'
        entry['class'] = extracted_label  # add label in dataframe in column 'class'

        tables.append(entry)  # combine entry with other entries for dataframe

    train_labels = pd.concat(tables, ignore_index=True)  # create dataframe from list of tables and reset index
    print(train_labels['class'].value_counts())  # get information on distribution of labels in dataframe

    return train_labels


def import_chest(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # set paths where training and test data can be found
    train_images = os.path.join(data_dir, "train")
    val_images = os.path.join(data_dir, "val")
    test_images = os.path.join(data_dir, "test")
    types = list(os.listdir(train_images))  # get unique labels (i.e. folders)

    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []

    for type_set in types:
        if type_set == ".DS_Store":
            continue
        else:
            for image_dir in [train_images, val_images, test_images]:
                sub_folder = os.path.join(image_dir, type_set)  # set path to images
                # get all files in folder ending with .jpg
                image = [os.path.join(sub_folder, f) for f in os.listdir(sub_folder) if f.endswith('.jpeg')]
                entry = pd.DataFrame(image, columns=['path'])  # add image in dataframe column 'path'
                entry['class'] = type_set  # add label in dataframe in column 'class'
                dataframe_entries.append(entry)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    return dataframe


def import_STL10(data_dir):
    """
    import file retrieved from: https://github.com/mttk/STL10/blob/master/stl10_input.py as suggested by STL-10 owners
    at Stanford on site https://cs.stanford.edu/~acoates/stl10/
    :param data_dir: directory where all data is stored (images and labels)
    :return: images and labels of STL-10 training dataset
    """
    # set paths to training or test images or labels
    train_img_path = os.path.join(data_dir, 'train_X.bin')
    train_label_path = os.path.join(data_dir, 'train_y.bin')
    test_img_path = os.path.join(data_dir, 'test_X.bin')
    test_label_path = os.path.join(data_dir, 'test_y.bin')

    with open(train_img_path, 'rb') as f:
        # read whole file in uint8 chunks
        all_train = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel, the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends on the input file, and this way numpy determines
        # the size on its own.
        train_images = np.reshape(all_train, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle if you will use a learning algorithm like CNN,
        # since they like their channels separated.
        train_images = np.transpose(train_images, (0, 3, 2, 1))

    with open(test_img_path, 'rb') as f:
        # read whole file in uint8 chunks
        all_test = np.fromfile(f, dtype=np.uint8)
        test_images = np.reshape(all_test, (-1, 3, 96, 96))
        test_images = np.transpose(test_images, (0, 3, 2, 1))

    with open(train_label_path, 'rb') as f:
        train_labels = np.fromfile(f, dtype=np.uint8)

    with open(test_label_path, 'rb') as f:
        test_labels = np.fromfile(f, dtype=np.uint8)

    # combine all data (to prevent accuracy issues due to very large test set)
    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    # decrement all labels with 1 to make sure the one-hot encoding happens right (takes max label + 1 as number of
    # classes so would become 11 classes instead of 10 if decrementing does not happen)
    labels_new = []
    for label in labels:
        label = label - 1
        labels_new.append(label)
    labels = np.array(labels_new)

    return images, labels


def import_textures_dtd(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # set paths where training and test data can be found
    types = list(os.listdir(data_dir))  # get all different labels

    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []

    for type_set in types:
        if type_set == ".DS_Store":
            continue
        else:
            sub_folder = os.path.join(data_dir, type_set)  # set path to images
            # get all files in folder ending with .jpg
            image = [os.path.join(sub_folder, f) for f in os.listdir(sub_folder) if f.endswith('.jpg')]
            entry = pd.DataFrame(image, columns=['path'])  # add image in dataframe column 'path'
            entry['class'] = type_set  # add label in dataframe in column 'class'
            dataframe_entries.append(entry)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    return dataframe


def import_PCAM(data_dir, source_data, target_data):
    """
    The .h5 files provided on https://github.com/basveeling/pcam have first been converted to numpy arrays in
    pcam_converter.py and saved locally as PNG-images. This function loads the png paths and labels in a dataframe.
    This was a workaround since using HDF5Matrix() from keras.utils gave errors when running Sacred.
    :param data_dir: directory where all data is stored (images and labels)
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .png
    images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]

    dataframe_entries = []  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=['path'])  # add img path to dataframe
        entry['class'] = img_path[-5:-4]  # add label in dataframe in column 'class'
        dataframe_entries.append(entry)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index

    # get pcam-middle subset
    if (target_data == 'pcam-middle') | (source_data == 'pcam-middle'):
        subset = dataframe.sample(n=100000, replace=False, random_state=2)
        print('Subset PCam-middle created', len(subset))
        print(subset['class'].value_counts())  # get information on distribution of labels in dataframe

        return subset

    # get pcam-small subset
    elif source_data == 'pcam-small':
        subset = dataframe.sample(n=10000, replace=False, random_state=22)
        print('Subset PCam-small created', len(subset))
        print(subset['class'].value_counts())  # get information on distribution of labels in dataframe

        return subset


def import_STI10(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: images and labels of own subset created from ImageNet in arrays
    """
    images = np.load(f'{data_dir}/all_imgs.npy', allow_pickle=True)  # import image array
    labels = np.load(f'{data_dir}/all_labels.npy', allow_pickle=True)  # import label array

    # reshape array into (len(images), 112, 112, 3)
    img_4d = []
    for img in images:
        img = cv2.resize(img, (112, 112))  # resize all images to 112x112
        img_4d.append(img)
    img_4d_arr = np.array(img_4d)  # convert images to array
    all_img = np.reshape(img_4d_arr, (len(img_4d_arr), 112, 112, 3))

    encoder = preprocessing.LabelEncoder()  # initiate encoder
    encoder = encoder.fit(labels)  # fit the encoder to the labels
    int_labels = encoder.transform(labels)  # transform the labels to integers

    return all_img, int_labels


def import_KimiaPath(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .tif
    images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]

    dataframe_entries = []  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        entry = pd.DataFrame([img_path], columns=['path'])  # add img path to dataframe
        # get label from image path, labels are of form LetterInteger (example A2 or A22)
        label = img_path[-7:-5]
        # some labels include double integer so need some preprocessing of the label: either remove / from label or
        # remove integer (i.e. last character of extracted label)
        if '/' in label:
            label = label.replace('/', '')
        else:
            label = label[:-1]
        entry['class'] = label  # append label to dataframe
        dataframe_entries.append(entry)  # combine entry with other entries for dataframe
    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    return dataframe


def collect_data(home, source_data, target_data):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :return: training, validation and test dataframe in case of pretraining, dataframe with all images and labels in
    case of transfer learning
    """
    if target_data is None:
        if source_data == 'isic':
            img_dir, label_dir = get_path(home, source_data)
            dataframe = import_ISIC(img_dir, label_dir)
        else:
            data_dir = get_path(home, source_data)

            if source_data == 'stl10':
                img, labels = import_STL10(data_dir)
            elif source_data == 'sti10':
                img, labels = import_STI10(data_dir)
            elif source_data == 'textures':
                dataframe = import_textures_dtd(data_dir)
            elif (source_data == 'pcam-middle') | (source_data == 'pcam-small'):
                dataframe = import_PCAM(data_dir, source_data, target_data)
            elif source_data == 'chest':
                dataframe = import_chest(data_dir)
            elif source_data == 'kimia':
                dataframe = import_KimiaPath(data_dir)

        if (source_data == 'stl10') | (source_data == 'sti10'):
            # split data in train-val-test set (train 80% - val 10% - test 10%)
            ten_percent = round(len(img) * 0.1)  # define 10% of whole dataset, pass on to split function
            X_train, X_test, y_train, y_test = train_test_split(img, labels, stratify=labels, shuffle=True,
                                                                random_state=2, test_size=ten_percent)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, shuffle=True,
                                                              random_state=2,
                                                              test_size=ten_percent)

            return X_train, X_val, X_test, y_train, y_val, y_test

        else:
            # split data in train-val-test set (train 80% - val 10% - test 10%)
            ten_percent = round(len(dataframe) * 0.1)  # define 10% of whole dataset, pass on to split function
            X_train, X_test, y_train, y_test = train_test_split(dataframe, dataframe['class'],
                                                                stratify=dataframe['class'],
                                                                test_size=ten_percent,
                                                                random_state=2, shuffle=True)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                              test_size=ten_percent,
                                                              random_state=2, shuffle=True)
            return X_train, X_val, X_test

    else:
        if target_data == 'isic':
            img_dir, label_dir = get_path(home, target_data)
            dataframe = import_ISIC(img_dir, label_dir)
            return dataframe

        else:
            data_dir = get_path(home, target_data)

            if target_data == 'chest':
                dataframe = import_chest(data_dir)
                return dataframe

            elif target_data == 'pcam-middle':
                dataframe = import_PCAM(data_dir, source_data, target_data)
                return dataframe
