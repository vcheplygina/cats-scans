# # %%
# from tensorflow.keras.preprocessing import image
# from src.io.data_import import import_textures_dtd, import_STI10, import_ISIC, import_STL10
# from keras.models import load_model
# from sklearn.model_selection import train_test_split
# from keras import models
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.applications.resnet50 import preprocess_input
# import cv2
# from PIL import Image
# #%%
# # X_train, X_val, X_test = import_textures_dtd(data_dir='/Users/IrmavandenBrandt/Downloads/Internship/dtd/images')
# # X_train, X_val, X_test, y_train, y_val, y_test = import_STI10(data_dir='/Users/IrmavandenBrandt/Downloads/Internship/sti10')
# #
# #
# # trained_model = load_model(f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained=sti10.h5')
# # # trained_model = load_model(f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained=textures.h5')
# #
# # resnet = trained_model.get_layer('resnet50')
# # print(resnet.summary())
# #%%
# # # get the flecked image 0093.jpg
# # path = '/Users/IrmavandenBrandt/Downloads/Internship/dtd/images/flecked/flecked_0093.jpg'
# #
# # for paths in X_train['path']:
# #     if paths == path:
# #         print('yes')
#
# # %%
# # show the activation maps on the first image of the test set
# # redefine model to output right after the first hidden layer
# first_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv1_conv').output)
# second_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv2_block1_1_conv').output)
# # %%
# # convert image to tensor and normalize
# # X_test_indexreset = X_test.reset_index(drop=True)
# # img = cv2.imread(X_train.loc[X_train['path'] == path]['path'][989])
# # img_tensor = image.img_to_array(img)
# # reshape image to 300x300x3
# # img_tensor = np.resize(img_tensor, (300, 300, 3))
# # img_tensor = preprocess_input(img_tensor)
# # img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
#
# # # get feature map for first hidden layer
# # activation = first_conv.predict(img_tensor_expanded)
# # # we have 64 filters in first conv layer so 64 different activation maps  (8 x 8)
# # index = 1
# # size1 = 8
# # size2 = 8
# # for _ in range(size1):
# #     for _ in range(size2):
# #         ax = plt.subplot(size1, size2, index)
# #         ax.set_xticks([])
# #         ax.set_yticks([])
# #         # get first filter activations
# #         plt.imshow(activation[0, :, :, index - 1])
# #         index += 1
# # plt.savefig("activations_freckled_0093_conv1", dpi=1000)
# # plt.show()
# #
# # # %%
# # img = Image.fromarray(img_tensor, 'RGB')
# # img.show()
# #
# #
# # #%%
# # # convert image to tensor and normalize
# # X_test_indexreset = X_test.reset_index(drop=True)
# # img = cv2.imread(X_test_indexreset.iloc[0]['path'])
# # img_tensor = image.img_to_array(img)
# # # reshape image to 300x300x3
# # img_tensor = np.resize(img_tensor, (300, 300, 3))
# # img_tensor = preprocess_input(img_tensor)
# # img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
# #
# # # get feature map for first hidden layer
# # activation = first_conv.predict(img_tensor_expanded)
# # # we have 64 filters in first conv layer so 64 different activation maps  (8 x 8)
# # index = 1
# # size1 = 8
# # size2 = 8
# # for _ in range(size1):
# #     for _ in range(size2):
# #         ax = plt.subplot(size1, size2, index)
# #         ax.set_xticks([])
# #         ax.set_yticks([])
# #         # get first filter activations
# #         plt.imshow(activation[0, :, :, index - 1])
# #         index += 1
# # plt.savefig("activations_wrinckled_0109", dpi=1000)
# # plt.show()
# #
# # # %%
# # img = Image.fromarray(img, 'RGB')
# # img.show()
#
# #%%
# X_train, X_val, X_test, y_train, y_val, y_test = import_STL10(
#     train_img_path='/Users/IrmavandenBrandt/Downloads/Internship/stl10_binary/train_X.bin',
# train_label_path='/Users/IrmavandenBrandt/Downloads/Internship/stl10_binary/train_y.bin',
# test_img_path='/Users/IrmavandenBrandt/Downloads/Internship/stl10_binary/test_X.bin',
# test_label_path='/Users/IrmavandenBrandt/Downloads/Internship/stl10_binary/test_y.bin')
# trained_model = load_model(f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained=stl10.h5')
# resnet = trained_model.get_layer('resnet50')
#
# # show the activation maps on the first image of the test set
# # redefine model to output right after the first hidden layer
# first_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv1_conv').output)
# second_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv2_block1_1_conv').output)
# #%%
# # preprocess image
# img = image.img_to_array(X_test[2])
# img_tensor = preprocess_input(img)
# img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
#
# # get feature map for first hidden layer
# activation = first_conv.predict(img_tensor_expanded)
# # we have 64 filters in first conv layer so 64 different activation maps  (8 x 8)
# index = 1
# size1 = 8
# size2 = 8
# for _ in range(size1):
#     for _ in range(size2):
#         ax = plt.subplot(size1, size2, index)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # get first filter activations
#         plt.imshow(activation[0, :, :, index - 1])
#         index += 1
# # plt.savefig("activations_wrinckled_0109", dpi=1000)
# plt.show()
# #%%
# I = cv2.cvtColor(X_test[0], cv2.COLOR_BGR2RGB)
# img = Image.fromarray(I, 'RGB')
# img.show()
#
# # #%%
# # dataframe = import_ISIC('/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018/ISIC2018_Task3_Training_Input',
# #                         '/Users/IrmavandenBrandt/Downloads/Internship/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
# # ten_percent = round(len(dataframe) * 0.1)
# # X_train, X_test, y_train, y_test = train_test_split(dataframe, dataframe['class'],
# #                                                     stratify=dataframe['class'], test_size=ten_percent,
# #                                                     random_state=2, shuffle=True)
# # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
# #                                                   stratify=y_train, test_size=ten_percent,
# #                                                  random_state=2, shuffle=True)
# # #%%
# # trained_model = load_model(f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained=isic.h5')
# # # trained_model = load_model(f'/Users/IrmavandenBrandt/Downloads/model_weights_resnet_pretrained=textures.h5')
# #
# # resnet = trained_model.get_layer('resnet50')
# #
# # # show the activation maps on the first image of the test set
# # # redefine model to output right after the first hidden layer
# # first_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv1_conv').output)
# # second_conv = models.Model(inputs=resnet.inputs, outputs=resnet.get_layer('conv2_block1_1_conv').output)
# # #%%
# # X_test_newindex = X_test.reset_index(drop=True)
# # img = cv2.imread(X_test_newindex.iloc[10]['path'])
# # img_tensor = image.img_to_array(img)
# # img_tensor = np.resize(img_tensor, (112, 112, 3))
# # img_tensor = preprocess_input(img_tensor)
# # img_tensor_expanded = np.expand_dims(img_tensor, axis=0)
# #
# # # get feature map for first hidden layer
# # activation = first_conv.predict(img_tensor_expanded)
# # # we have 64 filters in first conv layer so 64 different activation maps  (8 x 8)
# # index = 1
# # size1 = 8
# # size2 = 8
# # for _ in range(size1):
# #     for _ in range(size2):
# #         ax = plt.subplot(size1, size2, index)
# #         ax.set_xticks([])
# #         ax.set_yticks([])
# #         # get first filter activations
# #         plt.imshow(activation[0, :, :, index - 1])
# #         index += 1
# # # plt.savefig("activations_wrinckled_0109", dpi=1000)
# # plt.show()
# # #%%
# # I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # img = Image.fromarray(I, 'RGB')
# # img.show()
