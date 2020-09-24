# %%
from tensorflow.keras import optimizers, Sequential, losses
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB1
from keras.applications.resnet50 import ResNet50, preprocess_input
import os
from data_import import collect_data
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from requests_osf import upload_to_osf
from csv_writer import create_metrics_csv
import pandas as pd

# note: this is MACOS specific (to avoid OpenMP runtime error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_generators_dataframes(augment):
    """
    :param augment: boolean specifying whether to use data augmentation or not
    :return: training and validation generator, training and test dataset, and dictionary containing weights for all
    unique labels in the dataset
    """
    if augment:
        train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            preprocessing_function=preprocess_input)

    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # create validation generator
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    return train_datagen, valid_datagen


def create_model(y_col, learning_rate, img_length, img_width, color, dropout, imagenet, model_choice, dataframe):
    """
    :param y_col: column in dataframe containing the target labels
    :param learning_rate: learning rate used by optimizer
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param imagenet: boolean specifying whether or not to use pretrained imagenet weights in initialization model
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :param dataframe: dataframe containing paths of all images and the corresponding labels
    :return: compiled model (i.e. resnet or efficientnet)
    """
    # set input shape for model
    if color:
        input_shape = (img_length, img_width, 3)  # add 3 channels (i.e RGB) in case of color image
    else:
        input_shape = (img_length, img_width, 1)  # add 1 channels in case of gray image

    # compute number of nodes needed in prediction layer (i.e. number of unique classes)
    num_classes = len(list(dataframe[y_col].unique()))

    model = Sequential()  # initialize new model
    if model_choice == "efficientnet":
        if imagenet:
            # collect efficient net and exclude top layers
            efficient_net = EfficientNetB1(include_top=False, weights="imagenet", input_shape=input_shape)
        else:
            # collect efficient net and exclude top layers
            efficient_net = EfficientNetB1(include_top=False, weights=None, input_shape=input_shape)
        model.add(efficient_net)  # attach efficient net to new model
    elif model_choice == "resnet":
        if imagenet:
            # collect efficient net and exclude top layers
            resnet = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        else:
            # collect efficient net and exclude top layers
            resnet = ResNet50(include_top=False, weights=None, input_shape=input_shape)
        model.add(resnet)  # attach efficient net to new model
    # add new top layers to enable prediction for target dataset
    model.add(GlobalAveragePooling2D(name='gap'))
    model.add(Dropout(dropout, name='dropout_out'))
    model.add(Dense(num_classes, activation='softmax'))
    model.trainable = True  # set all layers in model to be trainable

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


def compute_class_weights(train_generator):
    """
    :param train_generator: training generator feeding batches of images to the model and storing class info
    :return: dictionary containing weights for all classes to balance them
    """
    # create balancing weights corresponding to the frequency of items in every class
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),
                                                      train_generator.classes)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    return class_weights


def run_model(isic, blood, x_col, y_col, augment, img_length, img_width, learning_rate, batch_size, epochs,
              color, dropout, imagenet, model_choice):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param learning_rate: learning rate used by optimizer
    :param batch_size: amount of images processed per batch
    :param epochs: number of epochs the model needs to run
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param imagenet: boolean specifying whether or not to use pretrained imagenet weights in initialization model
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :return: model and test generator needed for AUC calculation
    """

    # get generators
    train_datagen, valid_datagen = create_generators_dataframes(augment)

    # import data into function
    dataframe = collect_data(isic, blood)

    # initialize empty lists storing accuracy, loss and multi-class auc per fold
    acc_per_fold = []
    loss_per_fold = []
    auc_per_fold = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)  # create k-folds validator object with k=5

    fold_no = 1  # initialize fold counter

    for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):

        print(f'Starting fold {fold_no}')

        train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
        valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_data,
            x_col=x_col,
            y_col=y_col,
            target_size=(img_length, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            validate_filenames=False)

        validation_generator = valid_datagen.flow_from_dataframe(
            dataframe=valid_data,
            x_col=x_col,
            y_col=y_col,
            target_size=(img_length, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            validate_filenames=False,
            shuffle=False)

        model = create_model(y_col, learning_rate, img_length, img_width, color, dropout, imagenet, model_choice,
                             dataframe)  # create model

        class_weights = compute_class_weights(train_generator)  # get class weights to balance classes

        model.fit(train_generator,
                  steps_per_epoch=train_generator.samples // batch_size,
                  epochs=epochs,
                  class_weight=class_weights,
                  validation_data=validation_generator,
                  validation_steps=validation_generator.samples // batch_size,
                  )

        # compute loss and accuracy on validation set
        valid_loss, valid_acc = model.evaluate(validation_generator, verbose=1)
        print(f'Validation loss for fold {fold_no}:', valid_loss, f' and Validation accuracy for fold {fold_no}:',
              valid_acc)
        acc_per_fold.append(valid_acc)
        loss_per_fold.append(valid_loss)

        # save predictions in csv file
        predictions = model.predict(validation_generator)
        predictions_csv = np.savetxt(f'predictions_{model_choice}_fold{fold_no}.csv', predictions, delimiter=",")

        # compute OneVsRest multi-class macro AUC on the test set
        OneVsRest_auc = roc_auc_score(validation_generator.classes, predictions, multi_class='ovr', average='macro')
        print(f'Validation auc: {OneVsRest_auc}')
        auc_per_fold.append(OneVsRest_auc)

        if isic:
            # save predictions in osf or pycharm
            # np.savetxt(
            #     f'/Users/IrmavandenBrandt/PycharmProjects/cats-scans/predictions/predictions_{model_choice}_isic_fold{fold_no}.csv',
            #     predictions, delimiter=",")
            upload_to_osf(
                url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=predictions_{model_choice}_isic_fold{fold_no}.csv',
                file=predictions_csv,
                name=f'predictions_{model_choice}_isic_fold{fold_no}.csv')
            # write model to json file and upload to OSF
            upload_to_osf(
                url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_isic_fold{fold_no}.json',
                file=model.to_json(),
                name=f'model_{model_choice}_isic_fold{fold_no}.json')
            # save weights in h5 file and upload to OSF
            upload_to_osf(
                url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_isic_fold{fold_no}.h5',
                file=model.save(f'model_{model_choice}_isic_fold{fold_no}.h5'),
                name=f'model_{model_choice}_isic_fold{fold_no}.h5')
            print(f'Saved model and weights in OSF and finished fold {fold_no}')
        elif blood:
            # save predictions in osf or pycharm
            # np.savetxt(
            #     f'/Users/IrmavandenBrandt/PycharmProjects/cats-scans/predictions/predictions_{model_choice}_blood_fold{fold_no}.csv',
            #     predictions, delimiter=",")
            upload_to_osf(
                url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=predictions_{model_choice}_blood_fold{fold_no}.csv',
                file=predictions_csv,
                name=f'predictions_{model_choice}_blood_fold{fold_no}.csv')
            # write model to json file and upload to OSF
            upload_to_osf(
                url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_blood_fold{fold_no}.json',
                file=model.to_json(),
                name=f'model_{model_choice}_blood_fold{fold_no}.json')
            # save weights in h5 file and upload to OSF
            upload_to_osf(
                url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_blood_fold{fold_no}.h5',
                file=model.save(f'model_{model_choice}_blood_fold{fold_no}.h5'),
                name=f'model_{model_choice}_blood_fold{fold_no}.h5')
            print(f'Saved model and weights in OSF and finished fold {fold_no}')

        fold_no += 1  # increment fold counter to go to next fold

    # compute average scores for accuracy, loss and auc
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%, - AUC: {auc_per_fold[i]}')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
    print(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')

    # save model metrics in .csv file in project
    create_metrics_csv(model_choice, acc_per_fold, loss_per_fold, auc_per_fold)

    return acc_per_fold, loss_per_fold, auc_per_fold,

