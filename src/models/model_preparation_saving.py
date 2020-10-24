from sklearn.model_selection import StratifiedKFold, train_test_split
from ..io.requests_osf import upload_zip_to_osf
from ..io.data_import import collect_data
from .tf_generators_models_kfold import create_generators, compute_class_mode
import numpy as np
from keras.utils import to_categorical
from zipfile import ZipFile
import os


def prepare_model_source(augment, batch_size, source_data, home, target_data, img_length, img_width):
    """
    :param augment: boolean specifying whether to use data augmentation or not
    :param batch_size: amount of images processed per batch
    :param source_data: dataset used as source dataset
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param target_data: dataset used as target dataset
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :return: number of classes generators needed to create and compile the model
    """
    # get generators
    train_datagen, valid_datagen = create_generators(target_data, augment)

    # import data into function
    if (source_data == 'stl10') | (source_data == 'sti10'):
        X_train, X_val, X_test, y_train, y_val, y_test = collect_data(home, source_data)
        num_classes = len(np.unique(y_train))  # compute the number of unique classes in the dataset

        # convert labels to one-hot encoded labels after having counted the number of unique classes
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        # initiliaze generators fetching images from arrays
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

    else:
        if source_data == 'textures':
            train_dataframe, val_dataframe, test_dataframe = collect_data(home, source_data)

        elif (source_data == 'isic') | (source_data == 'pcam-middle') | (source_data == 'pcam-small') \
                | (source_data == 'chest'):
            dataframe = collect_data(home, source_data)

            # split data in train, val and test (80-10-10)
            ten_percent = round(len(dataframe) * 0.1)
            X_train, test_dataframe, y_train, y_test = train_test_split(dataframe, dataframe['class'],
                                                                        stratify=dataframe['class'],
                                                                        test_size=ten_percent,
                                                                        random_state=2, shuffle=True)
            train_dataframe, val_dataframe, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                                              test_size=ten_percent,
                                                                              random_state=2, shuffle=True)

        # get class model depending on the source data used in pretraining
        class_mode = compute_class_mode(source_data, target_data)

        num_classes = len(np.unique(train_dataframe['class']))  # compute the number of unique classes in the dataset

        # initiliaze generators fetching images from dataframe with image paths and labels
        train_generator = train_datagen.flow_from_dataframe(dataframe=train_dataframe,
                                                            x_col='path',
                                                            y_col='class',
                                                            target_size=(img_length, img_width),
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            class_mode=class_mode,
                                                            seed=2)

        validation_generator = valid_datagen.flow_from_dataframe(dataframe=val_dataframe,
                                                                 x_col='path',
                                                                 y_col='class',
                                                                 target_size=(img_length, img_width),
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 class_mode=class_mode,
                                                                 seed=2)

        test_generator = valid_datagen.flow_from_dataframe(dataframe=test_dataframe,
                                                           x_col='path',
                                                           y_col='class',
                                                           target_size=(img_length, img_width),
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           class_mode=class_mode,
                                                           seed=2)

    return num_classes, train_generator, validation_generator, test_generator


def prepare_model_target(home, target_data, source_data, x_col, y_col, augment, n_folds):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param target_data: dataset used as target dataset
    :param source_data: dataset used as source dataset
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param n_folds: amount of folds used in the n-fold cross validation
    :return: dataframe, number of classes in dataframe and column specifiers for generators, generators and
    stratified folds
    """

    # get generators
    train_datagen, valid_datagen = create_generators(target_data, augment)

    # create k-folds validator object with k=n_folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

    # collect the target data using the specified home path and the desired dataset name
    dataframe = collect_data(home, target_data)

    # compute number of nodes needed in prediction layer (i.e. number of unique classes)
    num_classes = len(list(dataframe[y_col].unique()))

    # compute class mode depending on target dataset
    class_mode = compute_class_mode(source_data, target_data)

    return num_classes, dataframe, skf, train_datagen, valid_datagen, x_col, y_col, class_mode


def save_pred_model(source_data, target_data, fold_no, model, predictions):
    """
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param fold_no: fold number that is currently used in the run
    :param model: compiled model
    :param predictions: class predictions obtained from the model on the target test set
    :return: saved predictions and model with weights
    """
    # save predictions first locally and then in osf
    np.savetxt(f'predictions_resnet_target={target_data}_source={source_data}_fold{fold_no}.csv',
               predictions, fmt='%1.3f', delimiter=",")
    # save model model_weights
    model.save(f'model_weights_resnet_target={target_data}_source={source_data}_fold{fold_no}.h5')
    print(f'Saved model and model_weights in zip and finished fold {fold_no}')


def create_upload_zip(n_folds, source_data, target_data):
    """
    :param n_folds: amount of folds used in the n-fold cross validation
    :param source_data: dataset used as src dataset
    :param target_data: dataset used as target dataset
    :return: zip-file uploaded on OSF containing predictions in case of target dataset and model with weights for both
    src and target
    """
    if target_data is None:
        with ZipFile(f'resnet_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            zip_object.write(f'model_weights_resnet_pretrained={source_data}.h5')

    else:
        with ZipFile(f'resnet_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            for i in range(1, n_folds + 1):
                # Add multiple files to the zip
                zip_object.write(f'predictions_resnet_target={target_data}_source={source_data}_fold{i}.csv')
                zip_object.write(f'model_weights_resnet_target={target_data}_source={source_data}_fold{i}.h5')

                # delete .csv and .h5 files from local memory
                os.remove(f'predictions_resnet_target={target_data}_source={source_data}_fold{i}.csv')
                os.remove(f'model_weights_resnet_target={target_data}_source={source_data}_fold{i}.h5')

    # upload zip to OSF
    upload_zip_to_osf(
        f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=resnet_target={target_data}_'
        f'source={source_data}.zip',
        f'resnet_target={target_data}_source={source_data}.zip',
        f'resnet_target={target_data}_source={source_data}.zip')
