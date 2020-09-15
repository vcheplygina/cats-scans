#%%
import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver
from tensorflow.keras import optimizers, models
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from data_import import import_melanoom
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# initialize experiment
ex = Experiment('pretrained_vgg')

# note: this is MACOS specific (to avoid OpenMP runtime error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@ex.config
def cfg():
    batch_size = 10
    num_epochs = 1
    learning_rate = 0.001


class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, _run):
        super().__init__()
        self._run = _run

    def on_epoch_end(self, _, logs):
        self._run.log_scalar("training.loss", logs.get('loss'))
        self._run.log_scalar("training.acc", logs.get('accuracy'))
        self._run.log_scalar("validation.loss", logs.get('val_loss'))
        self._run.log_scalar("validation.acc", logs.get('val_accuracy'))


@ex.automain
def run(_run, batch_size, num_epochs, learning_rate):
    # TODO: ask Veronika whether we can use the resized images (speeds up importing data)
    # import training data
    train_img, train_labels = import_melanoom(
        "/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Training_Data/resized",
        "/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Training_Part3_GroundTruth.csv",
        224, 224, norm=True, color=True, classes="three")

    # import validation data
    val_img, val_labels = import_melanoom(
        "/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Validation_Data/resized",
        "/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Validation_Part3_GroundTruth.csv",
        224, 224, norm=True, color=True, classes="three")

    # import test data
    test_img, test_labels = import_melanoom(
        "/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Test_v2_Data",
        "/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
        224, 224, norm=True, color=True, classes="three")

    # # %%
    # # show 1 image train images
    # # reading image
    # image = cv2.imread("/Users/IrmavandenBrandt/Downloads/Internship/ISIC-2017_Training_Data/resized/ISIC_0000004.jpg")
    # # displaying image
    # plt.imshow(image)
    # plt.show()

    # setting up the model
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3), classes=3)

    # preprocess input data before supplying it to the model
    X_train = preprocess_input(train_img)
    X_val = preprocess_input(val_img)
    X_test = preprocess_input(test_img)

    # add flatten and output layers to vgg16 model to get correct output
    VGG_model = vgg16.output
    VGG_model = Flatten()(VGG_model)
    VGG_model = Dense(128, activation='relu', name='dense_128')(VGG_model)
    predictions = Dense(3, activation='softmax', name='dense_3_class')(VGG_model)

    # get full model by combining the input from the vgg16 model and the output from our extension on the model
    model = models.Model(inputs=vgg16.input, outputs=predictions)

    model.compile(
        optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(X_train, train_labels,
              batch_size=256,
              epochs=1,
              validation_data=(X_val, val_labels),
              verbose=1)

    # evaluating the model
    test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=1)
    test_auc = roc_auc_score(test_labels, model.predict(X_test))
    print('Test loss:', test_loss, ' and Test accuracy:', test_acc)

    return test_auc