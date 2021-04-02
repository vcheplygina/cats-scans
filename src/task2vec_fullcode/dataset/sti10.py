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


class STI10Dataset(Dataset):
    """STI10 training dataset."""

    def __init__(self, root_dir, train, transform=None, rand_int=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = collect_data(home=root_dir, source_data='sti10',
                                                                      target_data=None)

        full_data_img = np.concatenate((X_train, X_val, X_test), axis=0)
        full_data_labels = np.concatenate((y_train, y_val, y_test), axis=0)
        np.random.seed(rand_int)
        indices = np.random.choice(range(len(full_data_img)), round(len(full_data_img) / 100), replace=False)
        sample_img = np.take(full_data_img, indices, axis=0)
        sample_labels = np.take(full_data_labels, indices)

        print(sample_img)
        self.sti10 = sample_img
        self.targets = sample_labels
        print(self.targets)

        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 10

    def __len__(self):
        return len(self.sti10)

    def __getitem__(self, idx):
        img_name = self.sti10[idx]
        image = Image.fromarray(img_name, 'RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
