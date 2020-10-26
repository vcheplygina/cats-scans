from tensorflow.keras.preprocessing import image
from src.io.data_import import collect_data
from keras.models import load_model
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import preprocess_input
import cv2
from PIL import Image


def collect_data_trainedmodel(target, source_data, target_data, home):
    if not target:
        # collect datasets
        if (source_data == 'isic') | (source_data == 'chest') | (source_data == 'pcam-middle') | \
                (source_data == 'pcam-small') | (source_data == 'textures'):
            X_train, X_val, X_test = collect_data(home, source_data, target_data)

        elif (source_data == 'stl10') | (source_data == 'sti10'):

            X_train, X_val, X_test, y_train, y_val, y_test = collect_data(home, source_data, target_data)

        # collect trained model and get resnet50 part
        trained_model = load_model(
            f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained={source_data}.h5')
        resnet = trained_model.get_layer('resnet50')

        # redefine model to output right after the first hidden layer
        first_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv1_conv').output)

        return X_train, X_val, X_test, first_conv


def visualize_featuremaps_firstconv(target, source_data, target_data, home, dataset, img_index, img_width, img_height):
    X_train, X_val, X_test, first_conv = collect_data_trainedmodel(target, source_data, target_data, home)

    print(X_train.loc[18]['path'])
    if (source_data == 'sti10') | (source_data == 'stl10'):
        if dataset == 'train':
            img = X_train[img_index]
        elif dataset == 'val':
            img = X_val[img_index]
        elif dataset == 'test':
            img = X_test[img_index]

    else:
        if dataset == 'train':
            # convert image to tensor and normalize
            X_train_resetindex = X_train.reset_index(drop=True)
            img = cv2.imread(X_train_resetindex.loc[img_index]['path'])

        elif dataset == 'val':
            # convert image to tensor and normalize
            X_val_resetindex = X_val.reset_index(drop=True)
            img = cv2.imread(X_val_resetindex.loc[img_index]['path'])

        elif dataset == 'test':
            # convert image to tensor and normalize
            X_test_resetindex = X_test.reset_index(drop=True)
            img = cv2.imread(X_test_resetindex.loc[img_index]['path'])

    # reshape image to 300x300x3
    img_tensor = image.img_to_array(img)
    img_tensor = np.resize(img_tensor, (img_width, img_height, 3))
    img_tensor = preprocess_input(img_tensor)
    img_tensor_expanded = np.expand_dims(img_tensor, axis=0)

    # get feature map for first conv layer
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
            plt.imshow(activation[0, :, :, index - 1])
            index += 1
    plt.savefig(f'activations_{source_data}_{dataset}_{img_index}', dpi=1000)
    plt.show()

    # show true image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb, 'RGB')
    img.show()


visualize_featuremaps_firstconv(target=False, source_data='textures', target_data=None,
                                home='/Users/IrmavandenBrandt/Downloads/Internship', dataset='train', img_index=18,
                                img_width=300, img_height=300)
