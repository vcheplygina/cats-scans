import keras

# importing the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalizing the images to 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# training configuration
batch_size = 64
num_epochs = 10
learning_rate = 0.01

# setting up the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compiling the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training the model
model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_data=(test_images, test_labels),
          verbose=2)

# evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test loss:', test_loss, ' and Test accuracy:', test_acc)