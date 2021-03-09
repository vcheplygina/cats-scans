from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from src.io.data_import import collect_data
from sklearn import preprocessing
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class STL10Dataset(Dataset):
    """STL10 training dataset."""

    def __init__(self, root_dir, train, transform=None, rand_int=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = collect_data(home=root_dir, source_data='stl10', target_data=None)

        full_data_img = np.concatenate((X_train, X_val, X_test), axis=None)
        full_data_labels = np.concatenate((y_train, y_val, y_test), axis=None)
        np.random.seed(rand_int)
        sample_img = np.random.choice(full_data_img, round(len(full_data_img) / 100), replace=False)
        sample_labels = np.random.choice(full_data_labels, round(len(full_data_labels) / 100), replace=False)

        self.stl10 = sample_img

        self.targets = sample_labels
        print(self.targets)

        self.root_dir = root_dir

        self.transform = transform
        self.num_classes = 10

    def __len__(self):
        return len(self.stl10)

    def __getitem__(self, idx):

        image = self.stl10[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
