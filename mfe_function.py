#%% Load dataset
import numpy as np
import keras
from scipy.stats import kurtosis, skew, entropy

# datasets = ['mnist']

for dataset in datasets:
    if dataset == 'mnist':
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



#%% Alterations for different dataset

# cifar10 = keras.datasets.cifar10
# (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# print(train_images.shape)

# cifar100 = keras.datasets.cifar100
# (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
# print(train_images.shape)

# b = 0.2989*train_images[0][0][0][0] + 0.5870*train_images[0][0][0][1] + 0.1140*train_images[0][0][0][2]
# print(train_images[0][0][0])
# train_images[0][0][0] = b
# print(train_images[0][0][0])

# if dataset = cifar10 or dataset = cifar100
#     for image in range(20):
#         for row in range(32):
#             for column in range(32):
#                 train_images[image][row][column] = 0.2989*train_images[image][row][column][0] + 0.5870*train_images[image][row][column][1] + 0.1140*train_images[image][row][column][2]

#%% General meta-feature extraction

    image_count = train_images.shape[0]
    image_size = train_images.shape[1]*train_images.shape[2]
    label_count = max(train_labels)

#%% Statistical meta-feature extraction

    meta_skew = []
    meta_kurtosis = []
    meta_entropy = []

    for index in range(20):                 # 20 moet zijn len(train_images)
        # Skew value
        meta_skew.append(skew(train_images[index], axis = None))
        # Kurtosis value
        meta_kurtosis.append(kurtosis(train_images[index], axis = None))
        # Entropy value
        meta_entropy.append(np.average(entropy(train_images[index], base = len(train_images[index].shape))))


#%% Determining the task specific features (binary)

#%% Task-specific meta-feature extraction

#%% Example vectors for 10 datasets
import random

mf1 = []
mf2 = []
mf3 = []
mf4 = []
mf5 = []
mf6 = []
mf7 = []
mf8 = []
mf9 = []
mf10 = []

vectors = [mf1, mf2, mf3, mf4, mf5, mf6, mf7, mf8, mf9, mf10]

for vector in vectors:
    for i in range(5):
        vector.append(random.randint(0,1))

#%% Similarity matrix calculation

sim_mat = np.zeros((len(vectors), len(vectors)))

# print(sim_mat[0][0])

for row in range(len(vectors)):
    for column in range(len(vectors)):
        euc_dst = np.linalg.norm(np.asarray(vectors[row]) - np.asarray(vectors[column]))
        sim_mat[row][column] = np.around(euc_dst, decimals=2)

print(sim_mat)

#%% Write to csv file

np.savetxt('similarity_matrix.csv', sim_mat)