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


class KimiaDataset(Dataset):
    """Kimia training dataset."""

    def __init__(self, root_dir, train, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X_train, X_val, X_test = collect_data(home=root_dir, source_data='kimia', target_data=None)
        if train:
            self.kimia = X_train
        else:
            self.kimia = X_test
        self.root_dir = root_dir
        self.transform = transform
        labelencoder = preprocessing.LabelEncoder()
        labelencoder.fit(self.kimia['class'])
        self.targets = labelencoder.transform(self.kimia['class'])
        print(self.targets)

    def __len__(self):
        return len(self.kimia)

    def __getitem__(self, idx):

        img_name = self.kimia.iloc[idx, 0]
        image = Image.open(img_name)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
