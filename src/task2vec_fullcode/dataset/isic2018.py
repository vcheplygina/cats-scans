from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from src.io.data_import import collect_data

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ISIC2018Dataset(Dataset):
    """ISIC 2018 training dataset."""

    def __init__(self, root_dir, train, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X_train, X_val, X_test = collect_data(home=root_dir, source_data='isic', target_data=None)
        if train:
            self.isic2018 = X_train
        else:
            self.isic2018 = X_test
        self.root_dir = root_dir
        self.transform = transform


        self.targets = self.isic2018['class'].astype(int)

    def __len__(self):
        return len(self.isic2018)

    def __getitem__(self, idx):

        img_name = self.isic2018.iloc[idx, 0]
        image = Image.open(img_name)
        target = int(self.isic2018.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, target
