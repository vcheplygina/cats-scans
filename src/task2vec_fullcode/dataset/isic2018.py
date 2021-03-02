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
from .dataset import ClassificationTaskDataset


class ISIC2018Dataset(ClassificationTaskDataset):
    """ISIC 2018 training dataset."""

    def __init__(self, root_dir, split='train', task_id=None, level=None,
                 metadata=None, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert isinstance(task_id, int)

        # create mapping between task ids and task labels
        if metadata is None:
            metadata = {}
        task_map = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}

        X_train, X_val, X_test = collect_data(home=root_dir, source_data='isic', target_data=None)

        # convert all labels in the datasets to the task ids
        X_train = X_train.replace({"class": task_map})
        print(X_train['class'].unique())

        if task_id:
            train_subset = X_train.loc[X_train['class'] == task_id]
            self.isic2018 = train_subset.reset_index(drop=True)
            print(self.isic2018)
        else:
            if task_id == 0:
                train_subset = X_train.loc[X_train['class'] == 0]
                self.isic2018 = train_subset.reset_index(drop=True)
                print(self.isic2018)
            else:
                self.isic2018 = X_train

        self.root_dir = root_dir

        targets = self.isic2018['class']

        if task_id:
            print(f'Embedding for task {task_id}')
            self.targets = targets[targets == task_id]
            print(self.targets)
        else:
            if task_id == 0:
                print(f'Embedding for task {task_id}')
                self.targets = targets[targets == task_id]
                print(self.targets)
            else:
                print('Domain embedding')
                self.targets = targets
                print(self.targets)

        self.meta_data = {}
        task_name = [key for key, value in task_map.items() if value == task_id]
        print(task_name[0])

        images_list = list(self.isic2018['path'])
        labels_list = list(self.targets)

        super(ISIC2018Dataset, self).__init__(images_list,
                                              labels_list,
                                              label_names=None,
                                              root=root_dir,
                                              task_id=task_id,
                                              task_name=task_name[0],
                                              metadata={},
                                              transform=transform,
                                              target_transform=None)

    def __len__(self):
        return len(self.isic2018)

    def __getitem__(self, idx):

        img_name = self.isic2018.iloc[idx, 0]
        image = Image.open(img_name)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target
