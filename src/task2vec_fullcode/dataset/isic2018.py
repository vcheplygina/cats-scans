from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from src.io.data_import import collect_data
from sklearn import preprocessing, utils
import numpy as np
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ISIC2018Dataset(Dataset):
    """ISIC 2018 training dataset."""

    def __init__(self, root_dir, train, transform=None, rand_int=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # collect the data
        X_train, X_val, X_test = collect_data(home=root_dir, source_data='isic', target_data=None)
        # combine all datasets in one dataset --> use this complete dataset to create all 100 subsets
        full_data = pd.concat([X_train, X_val, X_test])
        # create a sample of size len(dataset)/100 and use the random integer as random state
        labelencoder = preprocessing.LabelEncoder()
        labelencoder.fit(full_data['class'])
        full_data['class'] = labelencoder.transform(full_data['class'])
        full_data['freq'] = full_data.groupby('class')['class'].transform('count')
        sample = full_data.sample(n=round(len(full_data)/100), weights=full_data.groupby('class')['class'].transform('count'),
                                  random_state=rand_int, axis=None)
        sample = sample.reset_index(drop=True)

        self.isic2018 = sample
        self.root_dir = root_dir

        self.targets = self.isic2018['class']
        self.transform = transform
        self.num_classes = 7

    def __len__(self):
        return len(self.isic2018)

    def __getitem__(self, idx):

        img_name = self.isic2018.iloc[idx, 0]
        image = Image.open(img_name)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
