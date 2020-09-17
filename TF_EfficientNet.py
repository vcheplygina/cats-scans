# %%
from sacred import Experiment
from sacred.observers import MongoObserver
from tensorflow.keras import optimizers, models, Sequential, losses
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB1
import os
from data_import import collect_data

# note: this is MACOS specific (to avoid OpenMP runtime error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_generators_dataframes(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment, validation_split,
                                 img_length, img_width, batch_size):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param img_dir: directory where images are found
    :param label_dir: directory where labels are found
    :param test_size: split value used to split part of dataframe into test set
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param validation_split: fraction of images from training set used as validation set
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param batch_size: amount of images processed per batch
    :return: training and validation generator, training and test dataset
    """

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
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
            validation_split=validation_split)

    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=validation_split)

    # test this generator
    valid_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=validation_split)

    # import data into function
    df_train, df_test = collect_data(isic, blood, img_dir, label_dir, test_size)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col=x_col,
        y_col=y_col,
        target_size=(img_length, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        validate_filenames=False
    )

    validation_generator = valid_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col=x_col,
        y_col=y_col,
        target_size=(img_length, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        validate_filenames=False
    )

    return train_generator, validation_generator, df_train, df_test


def run_efficientnet(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment, validation_split, learning_rate,
                     img_length, img_width, batch_size, epochs, color, dropout, imagenet):
    """
    :param isic: boolean specifying whether data needed is ISIC data or not
    :param blood: boolean specifying whether data needed is blood data or not
    :param img_dir: directory where images are found
    :param label_dir: directory where labels are found
    :param test_size: split value used to split part of dataframe into test set
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param validation_split: fraction of images from training set used as validation set
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param batch_size: amount of images processed per batch
    :param learning_rate: learning rate used by optimizer
    :param epochs: number of epochs the model needs to run
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param imagenet: boolean specifying whether or not to use pretrained imagenet weights in initialization model
    :return: model and test generator needed for AUC calculation
    """

    # get generators and dataframes
    train_generator, validation_generator, df_train, df_test = create_generators_dataframes(isic, blood,
                                                                                            img_dir, label_dir,
                                                                                            test_size, x_col, y_col,
                                                                                            augment, validation_split,
                                                                                            img_length, img_width,
                                                                                            batch_size)

    # set input shape for model
    if color:
        input_shape = (img_length, img_width, 3)  # add 3 channels (i.e RGB) in case of color image
    else:
        input_shape = (img_length, img_width, 1)  # add 1 channels in case of gray image

    # compute number of nodes needed in prediction layer (i.e. number of unique classes)
    num_classes = len(list(df_train[y_col].unique()))  # todo: test this

    if imagenet:
        # collect efficient net and exclude top layers
        efficient_net = EfficientNetB1(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        # collect efficient net and exclude top layers
        efficient_net = EfficientNetB1(include_top=False, weights=None, input_shape=input_shape)
    model = Sequential()  # initialize new model
    model.add(efficient_net)  # attach efficient net to new model
    # add new top layers to enable prediction for target dataset
    model.add(GlobalAveragePooling2D(name='gap'))
    model.add(Dropout(dropout, name='dropout_out'))
    model.add(Dense(num_classes, activation='softmax'))
    model.trainable = True  # set all layers in model to be trainable

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              epochs=epochs,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // batch_size)  # todo: add early stopping

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_dataframe(df_test,
                                                      x_col="path",
                                                      y_col="class",
                                                      target_size=(img_length, img_width),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)

    # compute loss and accuracy on test set (NOTE: with .evaluate_generator() the accuracy is calculated per batch
    # and afterwards averaged, this can lead to lower accuracy scores than when using .evaluate()
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print('Test loss:', test_loss, ' and Test accuracy:', test_acc)

    return model, test_generator
