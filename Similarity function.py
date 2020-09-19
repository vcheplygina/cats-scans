import keras
import numpy as np
import matplotlib.pyplot as plt

# Importing the datasets

fashion_mnist = keras.datasets.fashion_mnist
(train_images1, train_labels1), (test_images1, test_labels1) = fashion_mnist.load_data()

mnist = keras.datasets.mnist
(train_images2, train_labels2), (test_images2, test_labels2) = mnist.load_data()

# D1 = np.array([3,7,5,1,8])
# D2 = np.array([6,3,1,2,7])

#%% Plot test

fig = plt.figure(figsize=(8, 8))
for i in range(1, 3):
    img = train_images1[i-1]
    fig.add_subplot(1, 2, i)
    plt.imshow(img)
plt.show()

# img = train_images1[0]
# plt.imshow(img)
# plt.show()

#%% Collecting dataset properties

image_count1 = train_images1.shape[0]
image_size1 = train_images1.shape[1]*train_images1.shape[2]
label_count1 = len(set(train_labels1))

image_count2 = train_images2.shape[0]
image_size2 = train_images2.shape[1]*train_images2.shape[2]
label_count2 = len(set(train_labels2))

#%% Collecting image properties

mean_pixel_list1= []

# for number in range(len(train_images1)):
for number in range(5):
    mean_pixel = np.average(np.average(train_images1[number])) # or calculate VARIANCE in mean
    mean_pixel_list1.append(mean_pixel)

avg_pix1 = np.average(mean_pixel_list1)

mean_pixel_list2= []

# for number in range(len(train_images2)):
for number in range(5):
    mean_pixel = np.average(np.average(train_images2[number])) # or calculate VARIANCE in mean
    mean_pixel_list2.append(mean_pixel)

avg_pix2 = np.average(mean_pixel_list2)

#SYMMETRY

#%%

