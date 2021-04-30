from sacred import Experiment
from src.models.model_preparation_saving import prepare_model_target, prepare_model_source, create_upload_zip, \
    save_pred_model
from src.models.tf_generators_models_kfold import create_model, compute_class_weights
import numpy as np
from src.evaluation.auc_evaluation import calculate_AUC
from neptunecontrib.monitoring.sacred import NeptuneObserver
from tensorflow.keras import callbacks
from numpy.random import seed
import tensorflow as tf
from src.io.access_keys import neptune_key

# initialize experiment name. NOTE: this should be updated with every new experiment
ex = Experiment('Resnet_pretrained=chest_target=chest')

ex.observers.append(NeptuneObserver(
    api_token=neptune_key,
    project_name='irmavdbrandt/cats-scans'))

seed(1)
tf.random.set_seed(2)


@ex.config
def cfg():
    """
    :return: parameter settings used in the experiment. NOTE: this should be updated with every new experiment
    """
    target = True
    # define src data
    source_data = "chest"
    # define target dataset
    target_data = "chest"
    x_col = "path"
    y_col = "class"
    augment = True
    k = 5
    img_length = 112
    img_width = 112
    learning_rate = 0.00001
    batch_size = 128
    epochs = 50
    color = True
    dropout = 0.5
    scheduler_bool = False
    home = '/data/ivdbrandt'


    # target = False
    # # define src data
    # source_data = "pcam-small"
    # # define target dataset
    # target_data = None
    # x_col = None
    # y_col = None
    # augment = True
    # n_folds = None
    # img_length = 96
    # img_width = 96
    # learning_rate = 0.00001
    # batch_size = 128
    # epochs = 50
    # color = True
    # dropout = 0.5
    # imagenet = False
    # scheduler_bool = False
    # home = '/data/ivdbrandt'


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
def run(_run, target, home, source_data, target_data, x_col, y_col, augment, k, img_length, img_width, learning_rate,
        batch_size, epochs, color, dropout, scheduler_bool):
    """
    :param _run:
    :param target: boolean specifying whether the run is for target data or src data
    :param home: part of path that is specific to user, e.g. /Users/..../
    :param source_data: dataset used as source dataset
    :param target_data: dataset used as target dataset
    :param x_col: column in dataframe containing the image paths
    :param y_col: column in dataframe containing the target labels
    :param augment: boolean specifying whether to use data augmentation or not
    :param k: amount of folds used in the k-fold cross validation
    :param img_length: target length of image in pixels
    :param img_width: target width of image in pixels
    :param learning_rate: learning rate used by optimizer
    :param batch_size: number of images processed in one batch
    :param epochs: number of iterations of the model per fold
    :param color: boolean specifying whether the images are in color or not
    :param dropout: fraction of nodes in layer that are deactivated
    :param scheduler_bool: boolean specifying whether learning rate scheduler is used
    :return: experiment
    """
    if scheduler_bool:
        # add learning rate scheduler in callbacks of model
        callbacks_settings = [MetricsLoggerCallback(_run),
                              callbacks.LearningRateScheduler(scheduler)]
    else:
        callbacks_settings = [MetricsLoggerCallback(_run)]

    if target:
        # collect all objects needed to prepare model for transfer learning
        dataframe, num_classes, x_col, y_col, class_mode, skf, train_gen, valid_gen = prepare_model_target(home,
                                                                                                           source_data,
                                                                                                           target_data,
                                                                                                           x_col,
                                                                                                           y_col,
                                                                                                           augment,
                                                                                                           k)

        # initialize empty lists storing accuracy, loss and multi-class auc per fold
        acc_per_fold = []
        loss_per_fold = []
        auc_per_fold = []

        fold_no = 1  # initialize fold counter

        for train_index, val_index in skf.split(np.zeros(len(dataframe)), y=dataframe[['class']]):
            print(f'Starting fold {fold_no}')

            train_data = dataframe.iloc[train_index]  # create training dataframe with indices from fold split
            valid_data = dataframe.iloc[val_index]  # create validation dataframe with indices from fold split

            train_generator = train_gen.flow_from_dataframe(dataframe=train_data,
                                                            x_col=x_col,
                                                            y_col=y_col,
                                                            target_size=(img_length, img_width),
                                                            batch_size=batch_size,
                                                            class_mode=class_mode,
                                                            seed=2,
                                                            validate_filenames=False)

            valid_generator = valid_gen.flow_from_dataframe(dataframe=valid_data,
                                                            x_col=x_col,
                                                            y_col=y_col,
                                                            target_size=(img_length, img_width),
                                                            batch_size=batch_size,
                                                            class_mode=class_mode,
                                                            validate_filenames=False,
                                                            seed=2,
                                                            shuffle=False)

            model = create_model(source_data, target_data, num_classes, learning_rate, img_length, img_width, color,
                                 dropout)  # create model using the compilation settings and image information

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
            print(f'Validation loss for fold {fold_no}: {valid_loss}', f' and Validation accuracy for fold {fold_no}: '
                                                                       f'{valid_acc}')
            acc_per_fold.append(valid_acc)
            loss_per_fold.append(valid_loss)

            predictions = model.predict(valid_generator)  # get predictions

            # calculate auc-score using y_true and predictions
            auc = calculate_AUC(target_data, valid_generator, predictions)
            auc_per_fold.append(auc)

            # save predictions and models_base in local memory
            save_pred_model(source_data, target_data, fold_no, model, predictions)

            fold_no += 1

        # create zip file with predictions and models_base and upload to OSF
        create_upload_zip(k, source_data, target_data)

        # compute average scores for accuracy, loss and auc
        print('Accuracy, loss and AUC per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%, - AUC: {auc_per_fold[i]}')
        print('Average accuracy, loss and AUC for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)})')
        print(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)})')

        return acc_per_fold, loss_per_fold, auc_per_fold

    else:
        # collect all objects needed to prepare model for pretraining
        num_classes, train_generator, valid_generator, test_generator = prepare_model_source(home,
                                                                                             source_data,
                                                                                             target_data,
                                                                                             augment,
                                                                                             batch_size,
                                                                                             img_length,
                                                                                             img_width)

        model = create_model(source_data, target_data, num_classes, learning_rate, img_length, img_width, color,
                             dropout)  # create model using the compilation settings and image information

        class_weights = compute_class_weights(train_generator.classes)  # get class model_weights to balance classes

        model.fit(train_generator,
                  epochs=epochs,
                  class_weight=class_weights,
                  validation_data=valid_generator,
                  callbacks=callbacks_settings)

        # compute loss and accuracy on validation set
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f'Test loss:', test_loss, f' and Test accuracy:', test_acc)

        # save model model_weights
        model.save(f'model_weights_resnet_pretrained={source_data}.h5')

        create_upload_zip(k, source_data, target_data)

        return test_loss, test_acc
