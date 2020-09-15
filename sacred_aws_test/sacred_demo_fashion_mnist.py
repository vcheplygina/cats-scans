#%%
import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver
from tensorflow.keras import optimizers, losses, datasets
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import os

# initialize experiment
ex = Experiment('fashion_mnist')
ex.observers.append(MongoObserver(url="mongodb+srv://Irma:MIA-Bas-Veronika@cats-scans.eqbh3.mongodb.net/sacred?retryWrites=true&w=majority"))

# note: this is MACOS specific (to avoid OpenMP runtime error)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@ex.config
def cfg():
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.01


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
    # importing the dataset
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # normalizing the images to 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # setting up the model
    model = Sequential()
    # add layers to model
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compiling the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # training the model
    model.fit(train_images, train_labels,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(test_images, test_labels),
              callbacks=[MetricsLoggerCallback(_run)],
              verbose=2)

    # evaluating the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('Test loss:', test_loss, ' and Test accuracy:', test_acc)

    return test_acc