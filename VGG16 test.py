from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# load the model
model = VGG16()
print(model.summary())

# load the image
image = load_img('mug.jpg', target_size = (224,224))

# convert the image pixels to a numpy array
image = img_to_array(image)

# reshape data for more images
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for VGG model
image = preprocess_input(image)

# predict the probability across all output classes
yhat = model.predict(image)

# convert the probabilities to class labels
label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability
label = label[0][0]

# print the classification
print(label[1], label[2]*100)