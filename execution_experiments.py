#%%
from sacred import Experiment
from sacred.observers import MongoObserver
from tf_generators_models_kfold import run_model
import tensorflow as tf

# initialize experiment
ex = Experiment('Efficientnet_ISIC_pretrained=Full_Imagenet')
ex.observers.append(MongoObserver(url="mongodb+srv://Irma:MIA-Bas-Veronika@cats-scans.eqbh3.mongodb.net/sacred"
                                      "?retryWrites=true&w=majority"))


@ex.config
def cfg():
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.00001
    img_length = 112
    img_width = 112
    drop_out = 0.2
    augment = True
    imagenet = True
    model_choice = 'efficientnet'


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
def run(_run, batch_size, num_epochs, learning_rate, img_length, img_width, drop_out, augment, imagenet, model_choice):
    """
    :param _run:
    :param batch_size:
    :param num_epochs:
    :param learning_rate:
    :param img_length:
    :param img_width:
    :param drop_out:
    :param augment:
    :param imagenet:
    :param model_choice:
    :return:
    """
    acc_per_fold, loss_per_fold, auc_per_fold = run_model(isic=True, blood=False, x_col="path", y_col="class",
                                                          augment=augment, img_length=img_length, img_width=img_width,
                                                          learning_rate=learning_rate, batch_size=batch_size,
                                                          epochs=num_epochs, color=True, dropout=drop_out,
                                                          imagenet=imagenet, model_choice=model_choice)

    return acc_per_fold, loss_per_fold, auc_per_fold
