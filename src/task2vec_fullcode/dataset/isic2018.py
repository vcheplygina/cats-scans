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


class ISIC2018Dataset(Dataset):
    """ISIC 2018 training dataset."""

    def __init__(self, root_dir, train, transform=None, task_id=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # create mapping between task ids and task labels
        task_map = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

        X_train, X_val, X_test = collect_data(home=root_dir, source_data='isic', target_data=None)

        # convert all labels in the datasets to the task ids
        X_train = X_train.replace({"class": task_map})
        X_test = X_test.replace({"class": task_map})

        if train:
            if task_id:
                self.isic2018 = X_train.loc[X_train['class'] == task_id]
            else:
                self.isic2018 = X_train
        else:
            if task_id:
                self.isic2018 = X_test.loc[X_test['class'] == task_id]
            else:
                self.isic2018 = X_test

        self.root_dir = root_dir
        self.transform = transform

        # labelencoder = preprocessing.LabelEncoder()
        # labelencoder.fit(self.isic2018['class'])
        # targets = labelencoder.transform(self.isic2018['class'])
        targets = self.isic2018['class']

        if task_id:
            print(f'Embedding for task {task_id}')
            self.targets = targets[targets == task_id]
            print(self.targets)
        else:
            print('Domain embedding')
            self.targets = targets
            print(self.targets)

        self.meta_data = {}
        self.task_name = [key for key, value in task_map.items() if value == task_id]

    def __len__(self):
        return len(self.isic2018)

    def __getitem__(self, idx):

        img_name = self.isic2018.iloc[idx, 0]
        image = Image.open(img_name)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
