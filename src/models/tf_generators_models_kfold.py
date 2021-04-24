from tensorflow.keras import optimizers, losses, models
from keras.models import load_model
from keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
import os
from sklearn.utils import class_weight
import numpy as np

# note: this is MACOS specific (to avoid OpenMP runtime error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_generators(dataset, augment):
    """
    :param dataset: dataset that is fed to generator
    :param augment: boolean specifying whether to use data augmentation or not
    :return: training and validation generator that preprocesses and augments images
    """
    # if (dataset == 'pcam-middle') | (dataset == 'pcam-small'):
    #     preprocessing = lambda x: x / 255.      # recommended preprocessing: https://github.com/basveeling/pcam
    # else:
    preprocessing = preprocess_input

    if augment:
        train_datagen = ImageDataGenerator(
            rotation_range=10,  # randomly rotate images in the range 0 to 180 (degrees)
            width_shift_range=0.1,  # randomly shift images horizontally, range as fraction of total width
            height_shift_range=0.1,  # randomly shift images vertically, range as fraction of total height
            horizontal_flip=True,  # randomly flip images
            preprocessing_function=preprocessing)

    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocessing)

    # create validation generator
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocessing)

    return train_datagen, valid_datagen


def create_model(source_data, target_data, num_classes, learning_rate, img_length, img_width, color, dropout):
    """
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param num_classes: number of unique classes in dataset
    :param learning_rate: learning rate used by optimizer
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :return: compiled model
    """
    # set input shape for model
    if color:
        input_shape = (img_length, img_width, 3)  # add 3 channels (i.e RGB) in case of color image
    else:
        input_shape = (img_length, img_width, 1)  # add 1 channels in case of gray image

    if (source_data != "imagenet") & (target_data is not None):
        # collect pretrained model on source data
        print(f'Loading model and weights from source data {source_data}')
        pretrained = load_model(
            f'model_weights_resnet_pretrained={source_data}.h5')
        # remove top layer that has been specialized on src dataset output
        if num_classes == 2:
            output_layer = Dense(1, activation='sigmoid')(pretrained.layers[-2].output)
            loss = losses.binary_crossentropy
        else:
            output_layer = Dense(num_classes, activation='softmax')(pretrained.layers[-2].output)
            loss = losses.categorical_crossentropy
        print(f'Prepared model for target dataset {target_data} with {source_data} weights')
        model = Model(inputs=pretrained.input, outputs=output_layer)
    else:
        model = models.Sequential()  # initialize new model
        if source_data == "imagenet":
            # collect resnet and exclude top layers
            resnet = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
            print(f'Prepared model for target dataset {target_data} with {source_data} weights')
        else:
            print(f'Initializing model for source dataset {source_data}')
            # collect resnet and exclude top layers
            resnet = ResNet50(include_top=False, weights=None, input_shape=input_shape)
        model.add(resnet)  # attach resnet to new model
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

    # compile model with accuracy as scoring metric
    model.compile(loss=loss,
                  optimizer=optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


def compute_class_weights(labels):
    """
    :param labels: array or column with labels
    :return: dictionary containing weights for all classes to balance them
    """
    # create balancing model_weights corresponding to the frequency of items in every class
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    return class_weights


def compute_class_mode(source_data, target_data):
    """
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :return: computes class mode depending on which source dataset is used in case of pretraining and using
    flow_from_dataframe or which target dataset is used in case of TF
    """
    if target_data is None:
        if (source_data == 'pcam-middle') | (source_data == 'pcam-small') | (source_data == 'chest'):
            class_mode = 'binary'
            return class_mode

        elif (source_data == 'isic') | (source_data == 'textures') | (source_data == 'kimia'):
            class_mode = 'categorical'
            return class_mode

    else:
        if target_data == "isic":
            class_mode = "categorical"
            return class_mode

        else:
            class_mode = "binary"  # binary class mode for chest and pcam
            return class_mode
