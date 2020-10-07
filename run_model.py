from sklearn.model_selection import StratifiedKFold
from requests_osf import upload_zip_to_osf
from data_import import collect_target_data, import_SLT10, import_textures_dtd
from tf_generators_models_kfold import create_generators_dataframes, compute_class_weights
import numpy as np
from keras.utils import to_categorical
from zipfile import ZipFile
import os
import pandas as pd


def run_model_source(augment, batch_size, source_data):
    """
    :param augment: boolean specifying whether to use data augmentation or not
    :param batch_size: amount of images processed per batch
    :param source_data: dataset used as source dataset
    :return: model and test generator needed for AUC calculation:
    """
    # get generators
    train_datagen, valid_datagen = create_generators_dataframes(augment)

    # import data into function
    if source_data == 'slt10':
        X_train, X_val, X_test, y_train, y_val, y_test = import_SLT10()
        num_classes = len(np.unique(y_train))  # compute the number of unique classes in the dataset
        class_weights = compute_class_weights(y_train)  # get class model_weights to balance classes

    elif source_data == 'textures':
        train_dataframe, val_dataframe, test_dataframe = import_textures_dtd()
        num_classes = len(np.unique(train_dataframe['class']))  # compute the number of unique classes in the dataset
        print(num_classes)
        class_weights = compute_class_weights(train_dataframe['class'])  # get class model_weights to balance classes

    if source_data == "textures":
        print("textures")
        train_generator = train_datagen.flow_from_dataframe(dataframe=train_dataframe,
                                                            x_col='path',
                                                            y_col='class',
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            class_mode="categorical",
                                                            seed=2)

        validation_generator = valid_datagen.flow_from_dataframe(dataframe=val_dataframe,
                                                                 x_col='path',
                                                                 y_col='class',
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 class_mode="categorical",
                                                                 seed=2)

        test_generator = valid_datagen.flow_from_dataframe(dataframe=test_dataframe,
                                                           x_col='path',
                                                           y_col='class',
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           class_mode="categorical",
                                                           seed=2)

    else:
        # convert labels to one-hot encoded labels
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        train_generator = train_datagen.flow(x=X_train,
                                             y=y_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             seed=2)

        validation_generator = valid_datagen.flow(x=X_val,
                                                  y=y_val,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  seed=2)

        test_generator = valid_datagen.flow(x=X_test,
                                            y=y_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            seed=2)

    return num_classes, train_generator, validation_generator, test_generator, class_weights


def run_model_target(target_data, x_col, y_col, augment, n_folds):
    """
    :param target_data: dataset used as target dataset
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param n_folds: amount of folds used in the n-fold cross validation
    :return: model and test generator needed for AUC calculation
    """

    # get generators
    train_datagen, valid_datagen = create_generators_dataframes(augment)

    # import data into function
    dataframe = collect_target_data(target_data)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)  # create k-folds validator object with k=5

    return dataframe, skf, train_datagen, valid_datagen, x_col, y_col


def save_pred_model(source_data, target_data, model_choice, fold_no, model, predictions):
    """
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :param fold_no: fold number that is currently used in the run
    :param model: compiled model
    :param predictions: class predictions obtained from the model on the target test set
    :return:
    """
    # save predictions first locally and then in osf
    np.savetxt(f'predictions_{model_choice}_target={target_data}_source={source_data}_fold{fold_no}.csv',
               predictions, fmt='%1.3f', delimiter=",")
    # save model model_weights
    model.save(f'model_weights_{model_choice}_target={target_data}_source={source_data}_fold{fold_no}.h5')
    print(f'Saved model and model_weights in zip and finished fold {fold_no}')


def create_upload_zip(n_folds, model_choice, source_data, target_data):
    """
    :param n_folds: amount of folds used in the n-fold cross validation
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :return:
    """
    if target_data is None:
        with ZipFile(f'{model_choice}_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            zip_object.write(f'model_weights_{model_choice}_pretrained={source_data}.h5')

            # # delete .csv and .h5 files from local memory
            # os.remove(f'model_weights_{model_choice}_pretrained={source_data}.h5')
    else:
        with ZipFile(f'{model_choice}_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            for i in range(1, n_folds + 1):
                # Add multiple files to the zip
                zip_object.write(f'predictions_{model_choice}_target={target_data}_source={source_data}_fold{i}.csv')
                zip_object.write(f'model_weights_{model_choice}_target={target_data}_source={source_data}_fold{i}.h5')

                # delete .csv and .h5 files from local memory
                os.remove(f'predictions_{model_choice}_target={target_data}_source={source_data}_fold{i}.csv')
                os.remove(f'model_weights_{model_choice}_target={target_data}_source={source_data}_fold{i}.h5')

    # upload zip to OSF
    upload_zip_to_osf(
        f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name={model_choice}_target={target_data}_source={source_data}.zip',
        f'{model_choice}_target={target_data}_source={source_data}.zip',
        f'{model_choice}_target={target_data}_source={source_data}.zip')
