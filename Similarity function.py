import keras
import numpy as np
import matplotlib.pyplot as plt

# Importing the datasets

fashion_mnist = keras.datasets.fashion_mnist
(train_images1, train_labels1), (test_images1, test_labels1) = fashion_mnist.load_data()

mnist = keras.datasets.mnist
(train_images2, train_labels2), (test_images2, test_labels2) = mnist.load_data()

D1 = np.array([3,0,5,0,8])
# D2 = np.array([6,3,1,2,7])

#%% Plotting images

fig = plt.figure(figsize=(3, 9))
for i in range(0, 3):
    img = train_images2[i]
    fig.add_subplot(3, 1, i+1)
    plt.imshow(img)
plt.show()

# img = train_images1[2]
# plt.imshow(img)
# plt.show()

#%% Collecting dataset properties (number of images, size of images and number of labels)

image_count1 = train_images1.shape[0]
image_size1 = train_images1.shape[1]*train_images1.shape[2]
label_count1 = len(set(train_labels1))

image_count2 = train_images2.shape[0]
image_size2 = train_images2.shape[1]*train_images2.shape[2]
label_count2 = len(set(train_labels2))

#%% Mean pixel value per image

mean_pixel_list1= []

# for number in range(len(train_images1)):
for imgnum in range(3):
    mean_pixel = np.average(np.average(train_images1[imgnum])) # or calculate VARIANCE in mean
    mean_pixel_list1.append(mean_pixel)

avg_pix1 = np.average(mean_pixel_list1)
std_pix1 = np.std(mean_pixel_list1)

mean_pixel_list2= []

# for number in range(len(train_images2)):
for imgnum in range(3):
    mean_pixel = np.average(np.average(train_images2[imgnum])) # or calculate VARIANCE in mean
    mean_pixel_list2.append(mean_pixel)

avg_pix2 = np.average(mean_pixel_list2)
std_pix2 = np.std(mean_pixel_list2)

#%% Ratio of foreground per image

fg_ratio1 = []

# for number in range(len(train_images1)):
for imgnum in range(3):
    num_zeros = (train_images1[imgnum] == 0).sum()
    ratio = 1-(num_zeros/(train_images1.shape[1]*train_images1.shape[2]))
    fg_ratio1.append(ratio)

avg_fg1 = np.average(fg_ratio1)
std_fg1 = np.std(fg_ratio1)

fg_ratio2 = []

# for number in range(len(train_images1)):
for imgnum in range(3):
    num_zeros = (train_images2[imgnum] == 0).sum()
    ratio = 1-(num_zeros/(train_images2.shape[1]*train_images2.shape[2]))
    fg_ratio2.append(ratio)
    
avg_fg2 = np.average(fg_ratio2)
std_fg2 = np.std(fg_ratio2)

#%% Create vector of properties

prop_vec1 = np.array([image_count1, image_size1, label_count1, std_pix1, std_fg1])
prop_vec2 = np.array([image_count2, image_size2, label_count2, std_pix2, std_fg2])

print(prop_vec2)

#%% Similarity measure of the vectors

euc_dst = np.linalg.norm(prop_vec1-prop_vec2)
print(euc_dist)

corco = np.corrcoef(prop_vec1, prop_vec2)[1, 0]
print(corco)
