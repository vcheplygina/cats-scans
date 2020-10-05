import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def import_ISIC():
    """
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # # local import paths
    # img_dir = "/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018/ISIC2018_Task3_Training_Input"
    # label_dir = "/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018/ISIC2018_Task3_Training_GroundTruth" \
    #             "/ISIC2018_Task3_Training_GroundTruth.csv"

    # server import paths
    img_dir = "/data/ivdbrandt/ISIC2018/ISIC2018_Task3_Training_Input"
    label_dir = "/data/ivdbrandt/ISIC2018/ISIC2018_Task3_Training_GroundTruth" \
                "/ISIC2018_Task3_Training_GroundTruth.csv"

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


def import_chest():
    """
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    data_dir = "/Users/IrmavandenBrandt/Downloads/Internship/chest_xray/chest_xray"
    # data_dir = "/data/ivdbrandt/chest_xray"

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


def import_SLT10():
    """
    import file retrieved from: https://github.com/mttk/STL10/blob/master/stl10_input.py as suggested by SLT10 owners
    at Stanford on site https://cs.stanford.edu/~acoates/stl10/
    :return: images and labels of SLT-10 training dataset
    """
    # path to the binary train file with image data
    TRAIN_DATA_PATH = '/Users/IrmavandenBrandt/Downloads/Internship/data_slt10/stl10_binary/train_X.bin'
    # path to the binary train file with labels
    TRAIN_LABEL_PATH = '/Users/IrmavandenBrandt/Downloads/Internship/data_slt10/stl10_binary/train_y.bin'
    # path to the binary train file with image data
    TEST_DATA_PATH = '/Users/IrmavandenBrandt/Downloads/Internship/data_slt10/stl10_binary/test_X.bin'
    # path to the binary train file with labels
    TEST_LABEL_PATH = '/Users/IrmavandenBrandt/Downloads/Internship/data_slt10/stl10_binary/test_y.bin'

    # # path to the binary train file with image data
    # TRAIN_DATA_PATH = '/data/ivdbrandt/stl10_binary/train_X.bin'
    # # path to the binary train file with labels
    # TRAIN_LABEL_PATH = '/data/ivdbrandt/stl10_binary/train_y.bin'
    # # path to the binary train file with image data
    # TEST_DATA_PATH = '/data/ivdbrandt/stl10_binary/test_X.bin'
    # # path to the binary train file with labels
    # TEST_LABEL_PATH = '/data/ivdbrandt/stl10_binary/test_y.bin'

    with open(TRAIN_DATA_PATH, 'rb') as f:
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

    with open(TEST_DATA_PATH, 'rb') as f:
        # read whole file in uint8 chunks
        all_test = np.fromfile(f, dtype=np.uint8)
        test_images = np.reshape(all_test, (-1, 3, 96, 96))
        test_images = np.transpose(test_images, (0, 3, 2, 1))

    with open(TRAIN_LABEL_PATH, 'rb') as f:
        train_labels = np.fromfile(f, dtype=np.uint8)

    with open(TEST_LABEL_PATH, 'rb') as f:
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def import_textures_dtd():
    """
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    data_dir = "/Users/IrmavandenBrandt/Downloads/Internship/dtd/images"
    # data_dir = "/data/ivdbrandt/dtd/images"

    # set paths where training and test data can be found
    types = list(os.listdir(data_dir))  # get all different labels

    # initiliaze empty lists that will store entries for train and test dataframes
    dataframe_entries = []

    for type_set in types:
        if type_set == ".DS_Store":
            continue
        else:
            for image_dir in data_dir:
                sub_folder = os.path.join(image_dir, type_set)  # set path to images
                image = [os.path.join(sub_folder, f) for f in os.listdir(sub_folder) if f.endswith('.jpg')]
                entry = pd.DataFrame(image, columns=['path'])  # add image in dataframe column 'path'
                entry['class'] = type_set  # add label in dataframe in column 'class'
                dataframe_entries.append(entry)  # combine entry with other entries for dataframe

    dataframe = pd.concat(dataframe_entries, ignore_index=True)  # create dataframe from list of tables and reset index
    print(dataframe['class'].value_counts())  # get information on distribution of labels in dataframe

    # split data in train-val-test set (train 1000, val 150 and test 150 per class)
    X_train, X_test, y_train, y_test = train_test_split(dataframe, dataframe['class'], stratify=dataframe['class'],
                                                        shuffle=True, random_state=2,
                                                        test_size=len(dataframe) / 0.1)  # take ~10% as test set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, shuffle=True, random_state=2,
                                                      test_size=len(dataframe) / 0.1)  # take ~10% as validation set

    return X_train, X_val, X_test, y_train, y_val, y_test


def collect_target_data(target_data):
    """
    :param target_data: dataset used as target dataset
    :return: dataframe containing image paths and labels
    """
    if target_data == 'isic':
        dataframe = import_ISIC()

    elif target_data == 'chest':
        dataframe = import_chest()

    return dataframe


