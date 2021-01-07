from sklearn.model_selection import StratifiedKFold
from src.io.requests_osf import upload_zip_to_osf
from src.io.data_import import collect_data
from .tf_generators_models_kfold import create_generators, compute_class_mode
import numpy as np
from keras.utils import to_categorical
from zipfile import ZipFile
import os


def prepare_model_source(home, source_data, target_data, augment, batch_size,  img_length, img_width):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param augment: boolean specifying whether to use data augmentation or not
    :param batch_size: amount of images processed per batch
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :return: number of classes in dataset and train, validation and test data-generators
    """
    # get generators
    train_datagen, valid_datagen = create_generators(target_data, augment)

    # import data into function
    if (source_data == 'stl10') | (source_data == 'sti10'):
        X_train, X_val, X_test, y_train, y_val, y_test = collect_data(home, source_data, target_data)
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
        # collect training, validation and testing datasets
        X_train, X_val, X_test = collect_data(home, source_data, target_data)

        # get class model depending on dataset used in pretraining, flow_from_dataframe needs classmode specification
        class_mode = compute_class_mode(source_data, target_data)

        num_classes = len(np.unique(X_train['class']))  # compute the number of unique classes in the dataset

        # initiliaze generators fetching images from dataframe with image paths and labels
        train_generator = train_datagen.flow_from_dataframe(dataframe=X_train,
                                                            x_col='path',
                                                            y_col='class',
                                                            target_size=(img_length, img_width),
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            class_mode=class_mode,
                                                            seed=2)

        validation_generator = valid_datagen.flow_from_dataframe(dataframe=X_val,
                                                                 x_col='path',
                                                                 y_col='class',
                                                                 target_size=(img_length, img_width),
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 class_mode=class_mode,
                                                                 seed=2)

        test_generator = valid_datagen.flow_from_dataframe(dataframe=X_test,
                                                           x_col='path',
                                                           y_col='class',
                                                           target_size=(img_length, img_width),
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           class_mode=class_mode,
                                                           seed=2)

    return num_classes, train_generator, validation_generator, test_generator


def prepare_model_target(home, source_data, target_data, x_col, y_col, augment, k):
    """
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param k: amount of folds used in the k-fold cross validation
    :return: dataframe with images and labels, number of classes in dataframe and column specifiers for data-generators,
    train and validation data-generators itself and stratified k-folds object
    """

    # get generators
    train_gen, valid_gen = create_generators(target_data, augment)

    # create k-folds validator object with k=n_folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2)

    # collect the target data using the specified home path and the desired dataset name
    dataframe = collect_data(home, source_data, target_data)

    # compute number of nodes needed in prediction layer (i.e. number of unique classes)
    num_classes = len(list(dataframe[y_col].unique()))

    # compute class mode depending on classification task associated with target dataset
    class_mode = compute_class_mode(source_data, target_data)

    return dataframe, num_classes, x_col, y_col, class_mode, skf, train_gen, valid_gen


def save_pred_model(source_data, target_data, fold_no, model, predictions):
    """
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param fold_no: fold number that is currently used in the run
    :param model: compiled model
    :param predictions: class predictions obtained from the model on the target test set
    :return: predictions and model with weights saved in local memory
    """
    # save predictions as csv file with predictions rounded to 3 numbers after decimal
    np.savetxt(f'predictions_resnet_target={target_data}_source={source_data}_fold{fold_no}.csv',
               predictions, fmt='%1.3f', delimiter=",")
    # save model and weights as .h5 file
    model.save(f'model_weights_resnet_target={target_data}_source={source_data}_fold{fold_no}.h5')
    print(f'Saved model and model_weights in zip and finished fold {fold_no}')


def create_upload_zip(k, source_data, target_data):
    """
    :param k: amount of folds used in the n-fold cross validation
    :param source_data: dataset used as src dataset
    :param target_data: dataset used as target dataset
    :return: zip-file uploaded on OSF containing predictions in case of target dataset and model with weights for both
    src and target
    """
    # in case of pretraining only the trained model and weights need to be stored, not the predictions
    if target_data is None:
        # write zipfile that contains model architecture and weights of pretrained model
        with ZipFile(f'resnet_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            zip_object.write(f'model_weights_resnet_pretrained={source_data}.h5')

    else:
        # write zipfile including model architecture, trained weights and predictions for every fold
        with ZipFile(f'resnet_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            for i in range(1, k + 1):
                zip_object.write(f'predictions_resnet_target={target_data}_source={source_data}_fold{i}.csv')
                zip_object.write(f'model_weights_resnet_target={target_data}_source={source_data}_fold{i}.h5')

                # delete .csv and .h5 files from local memory to save memory
                os.remove(f'predictions_resnet_target={target_data}_source={source_data}_fold{i}.csv')
                os.remove(f'model_weights_resnet_target={target_data}_source={source_data}_fold{i}.h5')

    # upload zipfile to OSF
    upload_zip_to_osf(
        f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=resnet_target={target_data}_'
        f'source={source_data}.zip',
        f'resnet_target={target_data}_source={source_data}.zip',
        f'resnet_target={target_data}_source={source_data}.zip')
