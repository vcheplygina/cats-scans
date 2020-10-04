import keras
import utils
from utils import meta_entropy, meta_skew, meta_kurtosis, meta_median, meta_std, meta_mean, meta_sparsity, meta_xy_axis

def feature_extraction(dataset = 'mnist', subset = 20):
    """calculates the similarity vector based on meta-features between datasets"""
    if dataset == 'mnist':
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        sub = train_images[:subset]
    else:
        return None

feature_extraction()
#%% Importing datasets

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


#%% Meta-feature extraction

data_entropy = []
data_skewness = []
data_kurtosis = []
data_median = []
data_std = []
data_mean = []
data_sparsity = []
data_xy_axis = []

for image in sub:
    data_entropy.append(round(meta_entropy(image, base = len(image.shape)), 2))
    data_skewness.append(round(meta_skew(image), 2))
    data_kurtosis.append(round(meta_kurtosis(image), 2))
    data_median.append(round(meta_median(image), 2))
    data_std.append(round(meta_std(image), 2))
    data_mean.append(round(meta_mean(image), 2))
    data_sparsity.append(round(meta_sparsity(image), 2))
    data_xy_axis.append(round(meta_xy_axis(image), 2))