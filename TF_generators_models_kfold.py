# %%
# from sacred import Experiment
# from sacred.observers import MongoObserver
from tensorflow.keras import optimizers, Sequential, losses
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB1
from keras.applications.resnet50 import ResNet50, preprocess_input
import os
from data_import import collect_data
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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


def run_model(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment, learning_rate,
              img_length, img_width, batch_size, epochs, color, dropout, imagenet, model_choice):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param img_dir: directory where images are found
    :param label_dir: directory where labels are found
    :param test_size: split value used to split part of dataframe into test set
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param batch_size: amount of images processed per batch
    :param learning_rate: learning rate used by optimizer
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
    dataframe = collect_data(isic, blood, img_dir, label_dir, test_size)

    # set input shape for model
    if color:
        input_shape = (img_length, img_width, 3)  # add 3 channels (i.e RGB) in case of color image
    else:
        input_shape = (img_length, img_width, 1)  # add 1 channels in case of gray image

    # compute number of nodes needed in prediction layer (i.e. number of unique classes)
    num_classes = len(list(dataframe[y_col].unique()))

    # initiliaze empty lists storing accuracy, loss and multi-class auc per fold
    acc_per_fold = []
    loss_per_fold = []
    auc_per_fold = []

    skf = StratifiedKFold(n_splits=5, random_state=2)  # create k-folds validator object with k=5

    fold_no = 1  # initialize fold counter

    for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):

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
            validate_filenames=False)

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

        # create balancing weights corresponding to the frequency of items in every class
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),
                                                          train_generator.classes)
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}
        print(class_weights)

        model.fit(train_generator,
                  steps_per_epoch=train_generator.samples // batch_size,
                  epochs=epochs,
                  class_weight=class_weights)

        # compute loss and accuracy on validation set
        valid_loss, valid_acc = model.evaluate(validation_generator, verbose=1)
        print(f'Test loss for fold {fold_no}:', valid_loss, f' and Test accuracy for fold {fold_no}:', valid_acc)
        acc_per_fold.append(valid_acc)
        loss_per_fold.append(valid_loss)

        # compute OneVsRest multi-class macro AUC on the test set
        OneVsRest_auc = roc_auc_score(validation_generator.classes, model.predict(validation_generator),
                                      multi_class='ovr',
                                      average='macro')
        auc_per_fold.append(OneVsRest_auc)

        # save model and weights in json files
        model_json = model.to_json()
        with open(f'/Users/IrmavandenBrandt/Downloads/Internship/models/model_{model_choice}_fold{fold_no}.json', "w") \
                as json_file:  # where to store models?
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(f'/Users/IrmavandenBrandt/Downloads/Internship/models/model_{model_choice}_fold{fold_no}.h5')
        # where to store weights?
        print("Saved model to disk")

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

    return acc_per_fold, loss_per_fold, auc_per_fold


acc_per_fold, loss_per_fold, auc_per_fold = run_model(isic=True, blood=False,
                                                      # img_dir="/Users/IrmavandenBrandt/Downloads/Internship/blood_data/9232-29380-bundle-archive"
                                                      #         "/dataset2-master/dataset2-master/images",
                                                      img_dir="/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018"
                                                              "/ISIC2018_Task3_Training_Input",
                                                      label_dir="/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018"
                                                                "/ISIC2018_Task3_Training_GroundTruth"
                                                                "/ISIC2018_Task3_Training_GroundTruth.csv",
                                                      # label_dir=None,
                                                      test_size=None, x_col="path", y_col="class",
                                                      augment=True, learning_rate=0.00001, img_length=60, img_width=80,
                                                      batch_size=128, epochs=10, color=True, dropout=0.2, imagenet=True,
                                                      model_choice="resnet")
