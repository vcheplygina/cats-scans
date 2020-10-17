from sacred import Experiment
from .models.run_model import run_model_target, run_model_source, create_upload_zip, save_pred_model
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from .models.tf_generators_models_kfold import create_model, compute_class_weights
import numpy as np
from neptunecontrib.monitoring.sacred import NeptuneObserver
from tensorflow.keras import callbacks

# initialize experiment name. NOTE: this should be updated with every new experiment
# ex = Experiment('Resnet_pretrained=imagenet_target=isic_test')
# ex = Experiment('Resnet_pretrained=pcam_target=chest')
ex = Experiment('Resnet_pretraining=STI10')


ex.observers.append(NeptuneObserver(
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjc4MGU5ZDUtMzk3Yy00YjE3LTliY2QtMThkMDJkZTMxNGMzIn0=",
    project_name='irmavdbrandt/cats-scans'))


@ex.config
def cfg():
    """
    :return: parameter settings used in the experiment. NOTE: this should be updated with every new experiment
    """
    # target = True
    # # define src data
    # source_data = "pcam"
    # # define target dataset
    # target_data = "chest"
    # x_col = "path"
    # y_col = "class"
    # augment = True
    # n_folds = 5
    # img_length = 112
    # img_width = 112
    # learning_rate = 0.000001
    # batch_size = 112
    # epochs = 50
    # color = True
    # dropout = 0.5
    # model_choice = "resnet"
    # scheduler_bool = True
    # home = '/data/ivdbrandt'

    target = False
    # define src data
    source_data = "sti10"
    # define target dataset
    target_data = None
    x_col = None
    y_col = None
    augment = True
    n_folds = None
    img_length = 112
    img_width = 112
    learning_rate = 0.0001
    batch_size = 128
    epochs = 50
    color = True
    dropout = 0.5
    imagenet = False
    model_choice = "resnet"
    scheduler_bool = False
    home = '/data/ivdbrandt'



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
    if epochs < 20:
        return learning_rate
    else:
        return learning_rate * 0.5


@ex.automain
def run(_run, target, target_data, source_data, x_col, y_col, augment, n_folds, img_length, img_width, learning_rate,
        batch_size, epochs, color, dropout, model_choice, scheduler_bool, home):
    """
    :param _run:
    :param target: boolean specifying whether the run is for target data or src data
    :param target_data: dataset used as target dataset
    :param source_data: dataset used as src dataset
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
    :param scheduler_bool: boolean specifying whether learning rate scheduler is used
    :param home: part of path that is specific to user, e.g. /Users/..../
    :return: experiment
    """

    if scheduler_bool:
        callbacks_settings = [MetricsLoggerCallback(_run),
                              callbacks.LearningRateScheduler(scheduler)]
    else:
        callbacks_settings = [MetricsLoggerCallback(_run)]

    if target:
        dataframe, skf, train_datagen, valid_datagen, x_col, y_col = run_model_target(home, target_data, x_col, y_col,
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

            if target_data == "isic":
                class_mode = "categorical"
            else:
                class_mode = "binary"

            train_generator = train_datagen.flow_from_dataframe(dataframe=train_data,
                                                                x_col=x_col,
                                                                y_col=y_col,
                                                                target_size=(img_length, img_width),
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                validate_filenames=False)

            valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid_data,
                                                                x_col=x_col,
                                                                y_col=y_col,
                                                                target_size=(img_length, img_width),
                                                                batch_size=batch_size,
                                                                class_mode=class_mode,
                                                                validate_filenames=False,
                                                                shuffle=False)

            # compute number of nodes needed in prediction layer (i.e. number of unique classes)
            num_classes = len(list(dataframe[y_col].unique()))

            model = create_model(target_data, learning_rate, img_length, img_width, color, dropout, source_data,
                                 model_choice, num_classes)  # create model

            class_weights = compute_class_weights(
                train_generator.classes)  # get class model_weights to balance classes

            model.fit(train_generator,
                      steps_per_epoch=train_generator.samples // batch_size,
                      epochs=epochs,
                      class_weight=class_weights,
                      validation_data=valid_generator,
                      validation_steps=valid_generator.samples // batch_size,
                      callbacks=callbacks_settings)

            # compute loss and accuracy on validation set
            valid_loss, valid_acc = model.evaluate(valid_generator, verbose=1)
            print(f'Validation loss for fold {fold_no}:', valid_loss,
                  f' and Validation accuracy for fold {fold_no}:',
                  valid_acc)
            acc_per_fold.append(valid_acc)
            loss_per_fold.append(valid_loss)

            predictions = model.predict(valid_generator)  # get predictions

            # compute OneVsRest multi-class macro AUC on the test set
            if target_data == "isic":
                OneVsRest_auc = roc_auc_score(valid_generator.classes, predictions, multi_class='ovr', average='macro')
            else:
                OneVsRest_auc = roc_auc_score(valid_generator.classes, predictions, average='macro')
            print(f'Validation auc: {OneVsRest_auc}')
            auc_per_fold.append(OneVsRest_auc)

            # save predictions and models_base in local memory
            save_pred_model(source_data, target_data, model_choice, fold_no, model, predictions)

            fold_no += 1

        # create zip file with predictions and models_base and upload to OSF
        create_upload_zip(n_folds, model_choice, source_data, target_data)

        # compute average scores for accuracy, loss and auc
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(
                f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%, - AUC: {auc_per_fold[i]}')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
        print(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')

        return acc_per_fold, loss_per_fold, auc_per_fold

    else:
        num_classes, train_generator, valid_generator, test_generator, class_weights = run_model_source(augment,
                                                                                                        batch_size,
                                                                                                        source_data,
                                                                                                        home,
                                                                                                        target_data,
                                                                                                        img_length,
                                                                                                        img_width)

        model = create_model(target_data, learning_rate, img_length, img_width, color, dropout, source_data,
                             model_choice, num_classes)  # create model

        model.fit(train_generator,
                  epochs=epochs,
                  class_weight=class_weights,
                  validation_data=valid_generator,
                  callbacks=callbacks_settings)

        # compute loss and accuracy on validation set
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f'Test loss:', test_loss, f' and Test accuracy:', test_acc)

        # save model model_weights
        model.save(f'model_weights_{model_choice}_pretrained={source_data}.h5')

        create_upload_zip(n_folds, model_choice, source_data, target_data)

        return test_loss, test_acc


# %%
import numpy as np
x = np.array([0.7816, 0.7833, 0.7820, 0.7816, 0.7797])
print(np.mean(x), np.std(x))
