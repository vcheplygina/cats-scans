# %%
from sacred import Experiment
# from sacred.observers import MongoObserver
from run_model import run_model_target, run_model_source, create_upload_zip, save_pred_model
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tf_generators_models_kfold import create_model, compute_class_weights
import numpy as np
from neptunecontrib.monitoring.sacred import NeptuneObserver
from tensorflow.keras import callbacks

# initialize experiment name. NOTE: this should be updated with every new experiment
ex = Experiment('Resnet_pretrained=Imagenet_source=Chest')
# ex = Experiment('Resnet_pretrained=Imagenet_source=Isic')
# ex = Experiment('Efficientnet_pretraining=SLT10')
ex.observers.append(NeptuneObserver(
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjc4MGU5ZDUtMzk3Yy00YjE3LTliY2QtMThkMDJkZTMxNGMzIn0=",
    project_name='irmavdbrandt/cats-scans'))


# create link with sacred MongoDB Atlas database
# ex.observers.append(MongoObserver(url="mongodb://localhost:27017/database"))
# url="mongodb+srv://Irma:MIA-Bas-Veronika@cats-scans.eqbh3.mongodb.net/sacred"
#                                   "?retryWrites=true&w=majority"))


@ex.config
def cfg():
    """
    :return: parameter settings used in the experiment. NOTE: this should be updated with every new experiment
    """
    target = True
    # define source data
    source_data = "imagenet"
    # define target dataset
    target_data = "chest"
    x_col = "path"
    y_col = "class"
    augment = True
    n_folds = 5
    img_length = 112
    img_width = 112
    learning_rate = 0.00001
    batch_size = 128
    epochs = 50
    color = True
    dropout = 0.2
    model_choice = "resnet"

    # target = False
    # # define source data
    # source_data = "slt10"
    # # define target dataset
    # target_data = None
    # x_col = None
    # y_col = None
    # augment = True
    # n_folds = None
    # img_length = 96
    # img_width = 96
    # learning_rate = 0.001  # with 0.0001 it goes too slow, with 0.001 it goes too fast (overfitting)
    # batch_size = 128
    # epochs = 50
    # color = True
    # dropout = 0.5  # with 0.4 and lr=0.001 still quick overfit
    # imagenet = False
    # model_choice = "resnet"


class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, _run):
        super().__init__()
        self._run = _run

    def on_epoch_end(self, _, logs):
        self._run.log_scalar("training.loss", logs.get('loss'))
        self._run.log_scalar("training.acc", logs.get('accuracy'))
        self._run.log_scalar("validation.loss", logs.get('val_loss'))
        self._run.log_scalar("validation.acc", logs.get('val_accuracy'))


def scheduler(epochs, learning_rate):
    if epochs < 30:
        return learning_rate
    else:
        return learning_rate * 0.1


@ex.automain
def run(_run, target, target_data, source_data, x_col, y_col, augment, n_folds, img_length, img_width, learning_rate,
        batch_size, epochs, color, dropout, model_choice):
    """
    :param _run:
    :param target: boolean specifying whether the run is for target data or source data
    :param target_data: dataset used as target dataset
    :param source_data: dataset used as source dataset
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param n_folds: amount of folds used in the n-fold cross validation
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param learning_rate: learning rate used by optimizer
    :param batch_size: number of images processed in one batch
    :param epochs: number of iterations of the model per fold
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param model_choice: model architecture to use for convolutional base (i.e. resnet or efficientnet)
    :return: experiment
    """

    # TODO: clean this code
    if target:
        dataframe, skf, train_datagen, valid_datagen, x_col, y_col = run_model_target(target_data, x_col, y_col,
                                                                                      augment, n_folds)

        # initialize empty lists storing accuracy, loss and multi-class auc per fold
        acc_per_fold = []
        loss_per_fold = []
        auc_per_fold = []

        fold_no = 1  # initialize fold counter

        for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
            print(f'Starting fold {fold_no}')

            train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
            valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split

            if target_data == "chest":
                train_generator = train_datagen.flow_from_dataframe(
                    dataframe=train_data,
                    x_col=x_col,
                    y_col=y_col,
                    target_size=(img_length, img_width),
                    batch_size=batch_size,
                    class_mode='binary',
                    validate_filenames=False)
            else:
                train_generator = train_datagen.flow_from_dataframe(
                    dataframe=train_data,
                    x_col=x_col,
                    y_col=y_col,
                    target_size=(img_length, img_width),
                    batch_size=batch_size,
                    class_mode='categorical',
                    validate_filenames=False)

            if target_data == "chest":
                validation_generator = valid_datagen.flow_from_dataframe(
                    dataframe=valid_data,
                    x_col=x_col,
                    y_col=y_col,
                    target_size=(img_length, img_width),
                    batch_size=batch_size,
                    class_mode='binary',
                    validate_filenames=False,
                    shuffle=False)
            else:
                validation_generator = valid_datagen.flow_from_dataframe(
                    dataframe=valid_data,
                    x_col=x_col,
                    y_col=y_col,
                    target_size=(img_length, img_width),
                    batch_size=batch_size,
                    class_mode='categorical',
                    validate_filenames=False,
                    shuffle=False)

            # compute number of nodes needed in prediction layer (i.e. number of unique classes)
            num_classes = len(list(dataframe[y_col].unique()))

            model = create_model(target_data, learning_rate, img_length, img_width, color, dropout, source_data,
                                 model_choice, num_classes)  # create model

            class_weights = compute_class_weights(train_generator.classes)  # get class model_weights to balance classes

            model.fit(train_generator,
                      steps_per_epoch=train_generator.samples // batch_size,
                      epochs=epochs,
                      class_weight=class_weights,
                      validation_data=validation_generator,
                      validation_steps=validation_generator.samples // batch_size,
                      callbacks=[MetricsLoggerCallback(_run)])

            # compute loss and accuracy on validation set
            valid_loss, valid_acc = model.evaluate(validation_generator, verbose=1)
            print(f'Validation loss for fold {fold_no}:', valid_loss, f' and Validation accuracy for fold {fold_no}:',
                  valid_acc)
            acc_per_fold.append(valid_acc)
            loss_per_fold.append(valid_loss)

            predictions = model.predict(validation_generator)  # get predictions

            # compute OneVsRest multi-class macro AUC on the test set
            if target_data == "chest":
                OneVsRest_auc = roc_auc_score(validation_generator.classes, predictions, average='macro')
            else:
                OneVsRest_auc = roc_auc_score(validation_generator.classes, predictions, multi_class='ovr',
                                              average='macro')
            print(f'Validation auc: {OneVsRest_auc}')
            auc_per_fold.append(OneVsRest_auc)

            # save predictions and models in local memory
            save_pred_model(source_data, target_data, model_choice, fold_no, model, predictions)

            fold_no += 1

        # create zip file with predictions and models and upload to OSF
        create_upload_zip(n_folds, model_choice, source_data, target_data)

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

    else:
        model, train_generator, validation_generator, test_generator, class_weights = run_model_source(augment,
                                                                                                       img_length,
                                                                                                       img_width,
                                                                                                       learning_rate,
                                                                                                       batch_size,
                                                                                                       epochs, color,
                                                                                                       dropout,
                                                                                                       source_data,
                                                                                                       model_choice)

        model.fit(train_generator,
                  epochs=epochs,
                  class_weight=class_weights,
                  validation_data=validation_generator,
                  callbacks=[MetricsLoggerCallback(_run),
                             callbacks.LearningRateScheduler(scheduler)])

        # compute loss and accuracy on validation set
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f'Test loss:', test_loss, f' and Test accuracy:', test_acc)

        # save model model_weights
        model.save(f'model_weights_{model_choice}_pretrained={source_data}.h5')

        create_upload_zip(n_folds, model_choice, source_data, target_data)

        return test_loss, test_acc
