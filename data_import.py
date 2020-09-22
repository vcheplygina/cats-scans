# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def import_ISIC(img_dir, label_dir):
    """
    :param img_dir: directory containing image paths
    :param label_dir: directory containing image labels
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


def import_blood(data_dir):
    """
    :param data_dir: directory where data is located
    :return: dataframe with image paths in column "path" and image labels in column "class"
    """
    # set paths where training and test data can be found
    train_images = os.path.join(data_dir, "TRAIN")
    test_images = os.path.join(data_dir, "TEST")
    types = list(os.listdir(train_images))  # get types of blood (e.g. labels)

    # initiliaze empty lists that will store entries for train and test dataframes
    train_tables = []
    test_tables = []

    for type_set in types:
        if type_set == '.DS_Store':
            continue
        else:
            train_sub_folder = os.path.join(train_images, type_set)  # set path to images
            train_image = [os.path.join(train_sub_folder, f) for f in os.listdir(train_sub_folder) if
                           f.endswith('.jpeg')]
            train_data_frame = pd.DataFrame(train_image, columns=['path'])  # add image in dataframe column 'path'
            train_data_frame['class'] = type_set  # add label in dataframe in column 'class'
            train_tables.append(train_data_frame)  # combine entry with other entries for dataframe

            test_sub_folder = os.path.join(test_images, type_set)  # set path to images
            test_image = [os.path.join(test_sub_folder, f) for f in os.listdir(test_sub_folder) if f.endswith('.jpeg')]
            test_data_frame = pd.DataFrame(test_image, columns=['path'])  # add image in dataframe column 'path'
            test_data_frame['class'] = type_set  # add label in dataframe in column 'class'
            test_tables.append(test_data_frame)  # combine entry with other entries for dataframe

    train_tables = pd.concat(train_tables, ignore_index=True)  # create dataframe from list of tables and reset index
    print(train_tables['class'].value_counts())  # get information on distribution of labels in dataframe

    test_tables = pd.concat(test_tables, ignore_index=True)  # create dataframe from list of tables and reset index
    print(test_tables['class'].value_counts())  # get information on distribution of labels in dataframe

    return train_tables, test_tables


# def collect_data(isic, blood, img_dir, label_dir, test_size):
#     """
#     :param isic: boolean specifying whether data needed is ISIC data or not
#     :param blood: boolean specifying whether data needed is blood data or not
#     :param img_dir: directory where images are found
#     :param label_dir: directory where labels are found
#     :param test_size: split value used to split part of dataframe into test set
#     :return: training, validation and test dataframe containing image paths and labels
#     """
#
#     if isic:
#         dataframe = import_ISIC(img_dir, label_dir)
#         # split the dataframe into stratified training and test set
#         X_train, X_test, y_train, y_test = train_test_split(dataframe['path'], dataframe['class'],
#                                                             stratify=dataframe['class'].values,
#                                                             shuffle=True, test_size=test_size, random_state=24)
#         df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
#         X_train, X_val, y_train, y_val = train_test_split(df_train['path'], df_train['class'],
#                                                             stratify=df_train['class'].values,
#                                                             shuffle=True, test_size=test_size, random_state=24)
#         # append labels to dataframes containing image paths and reset the index
#         df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
#         df_val = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
#         df_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
#
#     elif blood:
#         df_train, df_test = import_blood(img_dir)  # collect train and test dataframe
#         X_train, X_val, y_train, y_val = train_test_split(df_train['path'], df_train['class'],
#                                                           stratify=df_train['class'].values,
#                                                           shuffle=True, test_size=test_size, random_state=24)
#         df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
#         df_val = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
#
#     return df_train, df_val, df_test

def collect_data(isic, blood, img_dir, label_dir, test_size):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param img_dir: directory where images are found
    :param label_dir: directory where labels are found
    :param test_size: split value used to split part of dataframe into test set
    :return: training, validation and test dataframe containing image paths and labels
    """

    if isic:
        dataframe = import_ISIC(img_dir, label_dir)  # collect data

    elif blood:
        df_train, df_test = import_blood(img_dir)  # collect train and test dataframe
        dataframe = pd.concat([df_train, df_test])  # combine train and test dataframe in one dataframe

    return dataframe
