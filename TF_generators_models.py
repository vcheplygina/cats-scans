# # %%
# # from sacred import Experiment
# # from sacred.observers import MongoObserver
# from tensorflow.keras import optimizers, Sequential, losses
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from efficientnet.keras import EfficientNetB1
# from keras.applications.resnet50 import ResNet50, preprocess_input
# import os
# from data_import import collect_data
# from sklearn.utils import class_weight
# import numpy as np
# from sklearn.model_selection import KFold
#
# # note: this is MACOS specific (to avoid OpenMP runtime error)
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
#
# def create_generators_dataframes(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment,
#                                  img_length, img_width, batch_size, model_choice):
#     """
#     :param isic: boolean specifying whether data needed is ISIC data or not
#     :param blood: boolean specifying whether data needed is blood data or not
#     :param img_dir: directory where images are found
#     :param label_dir: directory where labels are found
#     :param test_size: split value used to split part of dataframe into test set
#     :param x_col: column in dataframe containing the image paths
#     :param y_col: column in dataframe containing the target labels
#     :param augment: boolean specifying whether to use data augmentation or not
#     :param img_length: target length of image in pixels
#     :param img_width: target width of image in pixels
#     :param batch_size: amount of images processed per batch
#     :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
#     :return: training and validation generator, training and test dataset, and dictionary containing weights for all
#     unique labels in the dataset
#     """
#     if augment:
#         train_datagen = ImageDataGenerator(
#                 featurewise_center=False,  # set input mean to 0 over the dataset
#                 samplewise_center=False,  # set each sample mean to 0
#                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
#                 samplewise_std_normalization=False,  # divide each input by its std
#                 zca_whitening=False,  # apply ZCA whitening
#                 rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#                 width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#                 height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#                 horizontal_flip=True,  # randomly flip images
#                 vertical_flip=False,
#                 preprocessing_function=preprocess_input)
#
#     else:
#         train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#
#     # create validation generator
#     valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#
#     # import data into function
#     df_train, df_val, df_test = collect_data(isic, blood, img_dir, label_dir, test_size)
#
#     train_generator = train_datagen.flow_from_dataframe(
#         dataframe=df_train,
#         x_col=x_col,
#         y_col=y_col,
#         target_size=(img_length, img_width),
#         batch_size=batch_size,
#         class_mode='categorical',
#         validate_filenames=False
#     )
#
#     validation_generator = valid_datagen.flow_from_dataframe(
#         dataframe=df_val,
#         x_col=x_col,
#         y_col=y_col,
#         target_size=(img_length, img_width),
#         batch_size=batch_size,
#         class_mode='categorical',
#         validate_filenames=False
#     )
#
#     # create balancing weights corresponding to the frequency of items in every class
#     class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),
#                                                       train_generator.classes)
#     class_weights = {i: class_weights[i] for i in range(len(class_weights))}
#     print(class_weights)
#
#     return train_generator, validation_generator, df_train, df_test, class_weights
#
#
# def run_model(isic, blood, img_dir, label_dir, test_size, x_col, y_col, augment, learning_rate,
#                      img_length, img_width, batch_size, epochs, color, dropout, imagenet, model_choice):
#     """
#     :param isic: boolean specifying whether data needed is ISIC data or not
#     :param blood: boolean specifying whether data needed is blood data or not
#     :param img_dir: directory where images are found
#     :param label_dir: directory where labels are found
#     :param test_size: split value used to split part of dataframe into test set
#     :param x_col: column in dataframe containing the image paths
#     :param y_col: column in dataframe containing the target labels
#     :param augment: boolean specifying whether to use data augmentation or not
#     :param img_length: target length of image in pixels
#     :param img_width: target width of image in pixels
#     :param batch_size: amount of images processed per batch
#     :param learning_rate: learning rate used by optimizer
#     :param epochs: number of epochs the model needs to run
#     :param color: boolean specifying whether the images are in color or not
#     :param dropout: fraction of nodes in layer that are deactivated
#     :param imagenet: boolean specifying whether or not to use pretrained imagenet weights in initialization model
#     :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
#     :return: model and test generator needed for AUC calculation
#     """
#
#     # get generators and dataframes
#     train_gen, validation_gen, df_train, df_test, class_weight_dict = create_generators_dataframes(isic,
#                                                                                                    blood,
#                                                                                                    img_dir,
#                                                                                                    label_dir,
#                                                                                                    test_size,
#                                                                                                    x_col,
#                                                                                                    y_col,
#                                                                                                    augment,
#                                                                                                    img_length,
#                                                                                                    img_width,
#                                                                                                    batch_size,
#                                                                                                    model_choice)
#
#     # set input shape for model
#     if color:
#         input_shape = (img_length, img_width, 3)  # add 3 channels (i.e RGB) in case of color image
#     else:
#         input_shape = (img_length, img_width, 1)  # add 1 channels in case of gray image
#
#     # compute number of nodes needed in prediction layer (i.e. number of unique classes)
#     num_classes = len(list(df_train[y_col].unique()))
#
#     model = Sequential()  # initialize new model
#
#     if model_choice == "efficientnet":
#         if imagenet:
#             # collect efficient net and exclude top layers
#             efficient_net = EfficientNetB1(include_top=False, weights="imagenet", input_shape=input_shape)
#         else:
#             # collect efficient net and exclude top layers
#             efficient_net = EfficientNetB1(include_top=False, weights=None, input_shape=input_shape)
#         model.add(efficient_net)  # attach efficient net to new model
#     elif model_choice == "resnet":
#         if imagenet:
#             # collect efficient net and exclude top layers
#             resnet = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
#         else:
#             # collect efficient net and exclude top layers
#             resnet = ResNet50(include_top=False, weights=None, input_shape=input_shape)
#         model.add(resnet)  # attach efficient net to new model
#     # add new top layers to enable prediction for target dataset
#     model.add(GlobalAveragePooling2D(name='gap'))
#     model.add(Dropout(dropout, name='dropout_out'))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.trainable = True  # set all layers in model to be trainable
#
#     model.compile(loss=losses.categorical_crossentropy,
#                   optimizer=optimizers.Adam(lr=learning_rate),
#                   metrics=['accuracy'])
#
#     model.fit(train_gen,
#               steps_per_epoch=train_gen.samples // batch_size,
#               epochs=epochs,
#               validation_data=validation_gen,
#               validation_steps=validation_gen.samples // batch_size,
#               class_weight=class_weight_dict)
#
#     # save model and weights in json files
#     model_json = model.to_json()
#     with open("/Users/IrmavandenBrandt/Downloads/Internship/models/model_{}.json".format(model_choice), "w") \
#             as json_file:  # where to store models?
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights("/Users/IrmavandenBrandt/Downloads/Internship/models/model_{}.h5".format(model_choice))
#     # where to store weights?
#     print("Saved model to disk")
#
#     # create test generator
#     test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
#
#     test_generator = test_datagen.flow_from_dataframe(df_test,
#                                                       x_col="path",
#                                                       y_col="class",
#                                                       target_size=(img_length, img_width),
#                                                       batch_size=batch_size,
#                                                       class_mode='categorical',
#                                                       shuffle=False)
#
#     # compute loss and accuracy on test set (NOTE: with .evaluate_generator() the accuracy is calculated per batch
#     # and afterwards averaged, this can lead to lower accuracy scores than when using .evaluate()
#     test_loss, test_acc = model.evaluate(test_generator, verbose=1)
#     print('Test loss:', test_loss, ' and Test accuracy:', test_acc)
#
#     return model, test_generator
