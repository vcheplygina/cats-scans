from tensorflow.keras import optimizers, losses, models
from keras.models import load_model
from keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB3
from keras.applications.resnet50 import ResNet50, preprocess_input
import os
from sklearn.utils import class_weight
import numpy as np

# note: this is MACOS specific (to avoid OpenMP runtime error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_generators_dataframes(target_data, augment):
    """
    :param target_data: dataset used as target dataset
    :param augment: boolean specifying whether to use data augmentation or not
    :return: training and validation generator, training and test dataset, and dictionary containing model_weights for
    all unique labels in the dataset
    """
    if target_data == 'pcam':
        preprocessing = lambda x: x / 255.
    else:
        preprocessing = preprocess_input

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
            preprocessing_function=preprocessing)

    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocessing)

    # create validation generator
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocessing)

    return train_datagen, valid_datagen


def create_model(target_data, learning_rate, img_length, img_width, color, dropout, source_data, model_choice,
                 num_classes):
    """
    :param target_data: dataset used as target dataset
    :param learning_rate: learning rate used by optimizer
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param source_data: dataset used as src dataset
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :param num_classes: number of unique classes in dataset
    :return: compiled model (i.e. resnet or efficientnet)
    """
    # set input shape for model
    if color:
        input_shape = (img_length, img_width, 3)  # add 3 channels (i.e RGB) in case of color image
    else:
        input_shape = (img_length, img_width, 1)  # add 1 channels in case of gray image

    if (source_data != "imagenet") & (target_data is not None):
        # collect pretrained efficientnet model on src data
        print(f'loading model and weights from source data {source_data} and for target data {target_data}')
        pretrained = load_model(
            f'model_weights_{model_choice}_pretrained={source_data}.h5')
        # remove top layer that has been specialized on src dataset output
        if num_classes == 2:
            output_layer = Dense(1, activation='sigmoid')(pretrained.layers[-2].output)
            loss = losses.binary_crossentropy
        else:
            output_layer = Dense(num_classes, activation='softmax')(pretrained.layers[-2].output)
            loss = losses.categorical_crossentropy
        model = Model(inputs=pretrained.input, outputs=output_layer)
    else:
        model = models.Sequential()  # initialize new model
        if model_choice == 'efficientnet':
            if source_data == "imagenet":
                # collect efficient net and exclude top layers
                efficient_net = EfficientNetB3(include_top=False, weights="imagenet", input_shape=input_shape)
            else:
                # collect efficient net and exclude top layers
                efficient_net = EfficientNetB3(include_top=False, weights=None, input_shape=input_shape)
            model.add(efficient_net)  # attach efficient net to new model
        elif model_choice == "resnet":
            if source_data == "imagenet":
                # collect efficient net and exclude top layers
                resnet = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
            else:
                # collect efficient net and exclude top layers
                resnet = ResNet50(include_top=False, weights=None, input_shape=input_shape)
            model.add(resnet)  # attach efficient net to new model
        # add new top layers to enable prediction for target dataset
        model.add(GlobalAveragePooling2D(name='gap'))
        model.add(Dropout(dropout, name='dropout_out'))
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = losses.binary_crossentropy
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss = losses.categorical_crossentropy

    model.trainable = True  # set all layers in model to be trainable

    model.compile(loss=loss,
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


def compute_class_weights(labels):
    """
    :param labels: dataset containing labels
    :return: dictionary containing model_weights for all classes to balance them
    """
    # create balancing model_weights corresponding to the frequency of items in every class
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    return class_weights
