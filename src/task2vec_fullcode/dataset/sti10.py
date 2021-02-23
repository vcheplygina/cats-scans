from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from src.io.data_import import collect_data
from sklearn import preprocessing

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class STI10Dataset(Dataset):
    """STI10 training dataset."""

    def __init__(self, root_dir, train, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = collect_data(home=root_dir, source_data='sti10', target_data=None)

        if train:
            self.sti10 = X_train
        else:
            self.sti10 = X_test
        self.root_dir = root_dir
        self.transform = transform
        self.targets = y_train
        print(self.targets)

    def __len__(self):
        return len(self.sti10)

    def __getitem__(self, idx):

        image = self.sti10[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
