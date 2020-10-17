import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from .data_paths import get_path
from sklearn import preprocessing


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
        data_frame = pd.DataFrame([img_path], columns=['path'])  # add img path to dataframe
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
        data_frame['class'] = extracted_label

        tables.append(data_frame)  # combine entry with other entries for dataframe

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
    types = list(os.listdir(train_images))  # get label (pneunomia or not)

    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []

    for type_set in types:
        if type_set == ".DS_Store":
            continue
        else:
            for image_dir in [train_images, val_images, test_images]:
                sub_folder = os.path.join(image_dir, type_set)  # set path to images
                image = [os.path.join(sub_folder, f) for f in os.listdir(sub_folder) if f.endswith('.jpeg')]
                entry = pd.DataFrame(image, columns=['path'])  # add image in dataframe column 'path'
                entry['class'] = type_set  # add label in dataframe in column 'class'
                dataframe_entries.append(entry)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    return dataframe


def import_STL10(train_img_path, train_label_path, test_img_path, test_label_path):
    """
    import file retrieved from: https://github.com/mttk/STL10/blob/master/stl10_input.py as suggested by STL-10 owners
    at Stanford on site https://cs.stanford.edu/~acoates/stl10/
    :param train_img_path: directory where training images are stored
    :param train_label_path: directory where training labels are stored
    :param test_img_path: directory where test images are stored
    :param test_label_path: directory where test labels are stored
    :return: images and labels of STL-10 training dataset
    """
    with open(train_img_path, 'rb') as f:
        # read whole file in uint8 chunks
        all_train = np.fromfile(f, dtype=np.uint8)
        print(all_train.shape)

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

    # split data in train-val-test set (train 1000, val 150 and test 150 per class)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify=labels, shuffle=True, random_state=2,
                                                        test_size=150 / 1300)  # take ~10% as test set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, shuffle=True, random_state=2,
                                                      test_size=150 / 1150)  # take ~10% as test set

    print(X_train.shape)


    return X_train, X_val, X_test, y_train, y_val, y_test


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
            image = [os.path.join(sub_folder, f) for f in os.listdir(sub_folder) if f.endswith('.jpg')]
            entry = pd.DataFrame(image, columns=['path'])  # add image in dataframe column 'path'
            entry['class'] = type_set  # add label in dataframe in column 'class'
            dataframe_entries.append(entry)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    # split data in train-val-test set (train 1000, val 150 and test 150 per class)
    X_train, X_test, y_train, y_test = train_test_split(dataframe, dataframe['class'], stratify=dataframe['class'],
                                                        shuffle=True, random_state=2,
                                                        test_size=round(len(dataframe) * 0.1))  # take ~10% as test set
    X_train, X_val, y_train, y_val = train_test_split(X_train, X_train['class'], stratify=X_train['class'],
                                                      shuffle=True, random_state=2,
                                                      test_size=round(len(dataframe) * 0.1))  # take ~10% as val set

    return X_train, X_val, X_test


def import_PCAM(data_dir):
    """
    The .h5 files provided on https://github.com/basveeling/pcam have first been converted to numpy arrays in
    pcam_converter.py and saved locally as png images. This function loads the png paths and labels in a dataframe.
    This was a workaround since using HDF5Matrix() from keras.utils gave errors when running Sacred.
    :param data_dir: directory where all data is stored (images and labels)
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # get image paths by selecting files from directory that end with .jpg
    images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]

    dataframe_entries = []  # initiliaze empty list that will store entries for dataframe

    for e, img_path in enumerate(images):
        data_frame = pd.DataFrame([img_path], columns=['path'])  # add img path to dataframe
        data_frame['class'] = img_path[-5:-4]  # add label in dataframe in column 'class'
        dataframe_entries.append(data_frame)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    # get subset of dataframe
    subset = dataframe.sample(n=100000, replace=False, random_state=2)
    print('subset created', len(subset))

    return subset


def import_STI10(data_dir):
    """
    :param data_dir: directory where all data is stored (images and labels)
    :return: images and labels of own subset created from ImageNet
    """
    images = np.load(f'{data_dir}/all_imgs.npy', allow_pickle=True)
    labels = np.load(f'{data_dir}/all_labels.npy', allow_pickle=True)

    all_imgs = np.reshape(images,

    # convert labels to integers
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(labels)
    print(list(encoder.classes_))
    int_labels = encoder.transform(labels)
    print(int_labels)

    # split data in train-val-test set (train 80% - val 10% - test 10%)
    ten_percent = round(0.1 * len(images))  # define 10% of whole dataset, pass on to split function
    X_train, X_test, y_train, y_test = train_test_split(images, int_labels, stratify=int_labels, shuffle=True, random_state=2,
                                                        test_size=ten_percent)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, shuffle=True, random_state=2,
                                                      test_size=ten_percent)
    print(X_train.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


def collect_data(home, target_data):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param target_data: dataset used as target dataset
    :return: dataframe containing image paths and labels
    """
    if target_data == 'isic':
        img_dir, label_dir = get_path(home, target_data)
        dataframe = import_ISIC(img_dir, label_dir)
        return dataframe

    elif target_data == 'chest':
        data_dir = get_path(home, target_data)
        dataframe = import_chest(data_dir)
        return dataframe

    elif target_data == 'stl10':
        train_img_path, train_label_path, test_img_path, test_label_path = get_path(home, target_data)
        X_train, X_val, X_test, y_train, y_val, y_test = import_STL10(train_img_path, train_label_path, test_img_path,
                                                                      test_label_path)
        return X_train, X_val, X_test, y_train, y_val, y_test

    elif target_data == 'textures':
        data_dir = get_path(home, target_data)
        X_train, X_val, X_test = import_textures_dtd(data_dir)
        return X_train, X_val, X_test

    elif target_data == 'pcam':
        data_dir = get_path(home, target_data)
        dataframe = import_PCAM(data_dir)
        return dataframe

    elif target_data == 'sti10':
        data_dir = get_path(home, target_data)
        X_train, X_val, X_test, y_train, y_val, y_test = import_STI10(data_dir)
        return X_train, X_val, X_test, y_train, y_val, y_test
