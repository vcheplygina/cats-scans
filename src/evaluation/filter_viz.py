#%%
from tensorflow.keras.preprocessing import image
from data_import import import_textures_dtd
from keras.models import load_model
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import preprocess_input
import cv2
from PIL import Image


X_train, X_val, X_test = import_textures_dtd()

trained_model = load_model(f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained=textures.h5')
resnet = trained_model.get_layer('resnet50')
print(resnet.summary())
#%%
# show the activation maps on the first image of the test set
# redefine model to output right after the first hidden layer
first_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv1_conv').output)

# convert image to tensor and normalize
X_test_indexreset = X_test.reset_index(drop=True)

#%%
img = cv2.imread(X_test_indexreset['path'][10])
img_tensor = image.img_to_array(img)
# reshape image to 300x300x3
img_tensor = np.resize(img_tensor, (300, 300, 3))
img_tensor = preprocess_input(img_tensor)
img_tensor_expanded = np.expand_dims(img_tensor, axis=0)

# get feature map for first hidden layer
activation = first_conv.predict(img_tensor_expanded)
# we have 64 filters in first conv layer so 64 different activation maps  (8 x 8)
index = 1
size1 = 8
size2 = 8
for _ in range(size1):
    for _ in range(size2):
        ax = plt.subplot(size1, size2, index)
        ax.set_xticks([])
        ax.set_yticks([])
        # get first filter activations
        plt.imshow(activation[0, :, :, index-1])
        index += 1
# plt.savefig("activations_firstlayer")
plt.show()

#%%
img = Image.fromarray(img, 'RGB')
img.show()