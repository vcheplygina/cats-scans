from sklearn.model_selection import StratifiedKFold
from requests_osf import upload_zip_to_osf
# from csv_writer import create_metrics_csv
from data_import import collect_data, import_SLT10
from tf_generators_models_kfold import create_model, create_generators_dataframes, compute_class_weights
import numpy as np
from keras.utils import to_categorical
from zipfile import ZipFile
import os


def run_model_source(augment, img_length, img_width, learning_rate, batch_size, epochs,
                     color, dropout, source_data, model_choice):
    """
    :param augment: boolean specifying whether to use data augmentation or not
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param learning_rate: learning rate used by optimizer
    :param batch_size: amount of images processed per batch
    :param epochs: number of epochs the model needs to run
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param source_data: dataset used as source dataset
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :return: model and test generator needed for AUC calculation:
    """
    # get generators
    train_datagen, valid_datagen = create_generators_dataframes(augment)

    # import data into function
    X_train, X_val, X_test, y_train, y_val, y_test = import_SLT10()
    print('length: ', len(y_train), ' frequencies: ', np.unique(y_train, return_counts=True))
    print('length: ', len(y_val), ' frequencies: ', np.unique(y_val, return_counts=True))
    print('length: ', len(y_test), ' frequencies: ', np.unique(y_test, return_counts=True))

    num_classes = len(np.unique(y_train))  # compute the number of unique classes in the dataset

    class_weights = compute_class_weights(y_train)  # get class model_weights to balance classes

    # convert labels to one-hot encoded labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    train_generator = train_datagen.flow(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=2)

    validation_generator = valid_datagen.flow(
        x=X_val,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=2)

    test_generator = valid_datagen.flow(
        x=X_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False,
        seed=2)

    model = create_model(learning_rate, img_length, img_width, color, dropout, source_data, model_choice,
                         num_classes)  # create model

    # model.fit(train_generator,
    #           # steps_per_epoch=train_generator.samples // batch_size,
    #           epochs=epochs,
    #           class_weight=class_weights,
    #           validation_data=validation_generator,
    #           # validation_steps=test_generator.samples // batch_size,
    #           )
    #
    # # compute loss and accuracy on validation set
    # test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    # print(f'Test loss:', test_loss, f' and Test accuracy:', test_acc)

    # upload_to_osf(
    #     url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_slt10.json',
    #     file=model.to_json(),
    #     name=f'model_{model_choice}_slt10.json')
    # # save model_weights in h5 file and upload to OSF
    # upload_to_osf(
    #     url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_slt10.h5',
    #     file=model.save(f'model_{model_choice}_slt10.h5'),
    #     name=f'model_{model_choice}_slt10.h5')

    return model, train_generator, validation_generator, test_generator, class_weights


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
    dataframe = collect_data(target_data)

    # # initialize empty lists storing accuracy, loss and multi-class auc per fold
    # acc_per_fold = []
    # loss_per_fold = []
    # auc_per_fold = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)  # create k-folds validator object with k=5

    # fold_no = 1  # initialize fold counter
    #
    # for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
    #
    #     print(f'Starting fold {fold_no}')
    #
    #     train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
    #     valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split
    #
    #     train_generator = train_datagen.flow_from_dataframe(
    #         dataframe=train_data,
    #         x_col=x_col,
    #         y_col=y_col,
    #         target_size=(img_length, img_width),
    #         batch_size=batch_size,
    #         class_mode='categorical',
    #         validate_filenames=False)
    #
    #     validation_generator = valid_datagen.flow_from_dataframe(
    #         dataframe=valid_data,
    #         x_col=x_col,
    #         y_col=y_col,
    #         target_size=(img_length, img_width),
    #         batch_size=batch_size,
    #         class_mode='categorical',
    #         validate_filenames=False,
    #         shuffle=False)
    #
    #     # compute number of nodes needed in prediction layer (i.e. number of unique classes)
    #     num_classes = len(list(dataframe[y_col].unique()))
    #
    #     model = create_model(learning_rate, img_length, img_width, color, dropout, imagenet, model_choice,
    #                          num_classes)  # create model
    #
    #     class_weights = compute_class_weights(train_generator.classes)  # get class model_weights to balance classes
    #
    #     model.fit(train_generator,
    #               steps_per_epoch=train_generator.samples // batch_size,
    #               epochs=epochs,
    #               class_weight=class_weights,
    #               validation_data=validation_generator,
    #               validation_steps=validation_generator.samples // batch_size
    #               )
    #
    #     # compute loss and accuracy on validation set
    #     valid_loss, valid_acc = model.evaluate(validation_generator, verbose=1)
    #     print(f'Validation loss for fold {fold_no}:', valid_loss, f' and Validation accuracy for fold {fold_no}:',
    #           valid_acc)
    #     acc_per_fold.append(valid_acc)
    #     loss_per_fold.append(valid_loss)
    #
    #     # get predictions
    #     predictions = model.predict(validation_generator)
    #
    #     # compute OneVsRest multi-class macro AUC on the test set
    #     OneVsRest_auc = roc_auc_score(validation_generator.classes, predictions, multi_class='ovr', average='macro')
    #     print(f'Validation auc: {OneVsRest_auc}')
    #     auc_per_fold.append(OneVsRest_auc)

    #     if isic:
    #         # save predictions first locally and then in osf
    #         np.savetxt(f'predictions/predictions_{model_choice}_isic_fold{fold_no}_test.csv',
    #                    predictions, fmt='%1.3f', delimiter=",")
    #         # upload_csv_to_osf(
    #         #     url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=predictions_{model_choice}_isic_fold{fold_no}.csv',
    #         #     file_name=f'predictions/predictions_{model_choice}_isic_fold{fold_no}.csv',
    #         #     name=f'predictions_{model_choice}_isic_fold{fold_no}.csv')
    #         # save model model_weights
    #         model.save(f'model_weights/model_weights_{model_choice}_isic_fold{fold_no}_test.h5')
    #         print(f'Saved model and model_weights in zip and finished fold {fold_no}')
    #     elif blood:  # todo: change
    #         continue
    #
    #     fold_no += 1  # increment fold counter to go to next fold
    #
    # with ZipFile(f'{model_choice}_isic_Imagenet={imagenet}.zip', 'w') as zip_object:
    #     for i in range(1, 6):
    #         # Add multiple files to the zip
    #         zip_object.write(f'predictions/predictions_{model_choice}_isic_fold{i}_test.csv')
    #         zip_object.write(f'model_weights/model_weights_{model_choice}_isic_fold{i}_test.h5')
    #
    # # upload zip to OSF
    # upload_zip_to_osf(f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name={model_choice}_isic_Imagenet={imagenet}.zip',
    #                   f'{model_choice}_isic_Imagenet={imagenet}.zip',
    #                   f'{model_choice}_isic_Imagenet={imagenet}.zip')

    # write model to json file and upload to OSF
    # if isic:
    #     # upload_model_to_osf(
    #     #     url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_isic.json',
    #     #     file=model.to_json(),
    #     #     name=f'model_{model_choice}_isic.json')
    #     zip_object.write(model.to_json())
    # elif blood:
    #     upload_model_to_osf(
    #         url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=model_{model_choice}_blood.json',
    #         file=model.to_json(),
    #         name=f'model_{model_choice}_blood_.json')

    # # compute average scores for accuracy, loss and auc
    # print('Score per fold')
    # for i in range(0, len(acc_per_fold)):
    #     print('------------------------------------------------------------------------')
    #     print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%, - AUC: {auc_per_fold[i]}')
    # print('Average scores for all folds:')
    # print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    # print(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
    # print(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')
    #
    # # save model metrics in .csv file in project and store in OSF
    # if isic:
    #     create_metrics_csv(isic, blood, model_choice, acc_per_fold, loss_per_fold, auc_per_fold)
    #     # upload_csv_to_osf(
    #     #     url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=metrics_{model_choice}_isic_test.csv',
    #     #     file_name=f'model_results/metrics_{model_choice}_isic.csv',
    #     #     name=f'metrics_{model_choice}_isic.csv')
    # elif blood:
    #     create_metrics_csv(isic, blood, model_choice, acc_per_fold, loss_per_fold, auc_per_fold)
        # upload_csv_to_osf(
        #     url=f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name=metrics_{model_choice}_blood.csv',
        #     file_name=f'model_results/metrics_{model_choice}_blood.csv',
        #     name=f'metrics_{model_choice}_blood.csv')

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

            # delete .csv and .h5 files from local memory
            os.remove(f'model_weights_{model_choice}_pretrained={source_data}.h5')
    else:
        with ZipFile(f'{model_choice}_target={target_data}_source={source_data}.zip', 'w') as zip_object:
            for i in range(1, n_folds+1):
                # Add multiple files to the zip
                zip_object.write(f'predictions_{model_choice}_target={target_data}_source={source_data}_fold{i}.csv')
                zip_object.write(f'model_weights_{model_choice}_target={target_data}_source={source_data}_fold{i}.h5')

                # delete .csv and .h5 files from local memory
                os.remove(f'predictions_{model_choice}_target={target_data}_source={source_data}_fold{i}.csv')
                os.remove(f'model_weights_{model_choice}_target={target_data}_source={source_data}_fold{i}.h5')

    # upload zip to OSF
    upload_zip_to_osf(f'https://files.osf.io/v1/resources/x2fpg/providers/osfstorage/?kind=file&name={model_choice}_target={target_data}_source={source_data}.zip',
                      f'{model_choice}_target={target_data}_source={source_data}.zip',
                      f'{model_choice}_target={target_data}_source={source_data}.zip')

# # %%
# acc_per_fold, loss_per_fold, auc_per_fold = run_model_target(isic=True, blood=False, x_col="path", y_col="class",
#                                                              augment=True, img_length=32, img_width=32,
#                                                              learning_rate=0.00001, batch_size=128,
#                                                              epochs=1, color=True, dropout=0.2,
#                                                              imagenet=True, model_choice="efficientnet")

# %%
# test_loss, test_acc = run_model_source(augment=False, img_length=96, img_width=96,
#                                        learning_rate=0.0001, batch_size=128,
#                                        epochs=10, color=True, dropout=0.2,
#                                        imagenet=False, model_choice="efficientnet")
