from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from src.io.data_import import collect_data
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class PCAMDataset(Dataset):
    """PCam dataset."""

    def __init__(self, root_dir, train, transform=None, rand_int=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X_train, X_val, X_test = collect_data(home=root_dir, source_data='pcam-middle', target_data=None)
        # combine all datasets in one dataset --> use this complete dataset to create all 100 subsets
        full_data = pd.concat([X_train, X_val, X_test])
        sample = full_data.sample(n=round(len(full_data) / 100),
                                  weights=full_data.groupby('class')['class'].transform('count'),
                                  random_state=rand_int, axis=None)
        sample = sample.reset_index(drop=True)

        self.PCAM = sample
        self.root_dir = root_dir
        self.targets = self.PCAM['class'].astype(int)
        self.transform = transform
        self.num_classes = 2
        print(self.targets)

    def __len__(self):
        return len(self.PCAM)

    def __getitem__(self, idx):

        img_name = self.PCAM.iloc[idx, 0]
        image = Image.open(img_name)
        target = int(self.PCAM.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, target
