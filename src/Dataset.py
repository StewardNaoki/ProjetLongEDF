import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms


class LP_data(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, csv_file_name, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file_name)
        self.transform = transform
        # self.IMG_SIZE = 64

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        A = self.data_frame["A"].iloc[idx]
        A = eval(A)

        B = self.data_frame["B"].iloc[idx]
        B = eval(B)

        C = self.data_frame["C"].iloc[idx]
        C = eval(C)

        # print(A)
        # print(B)
        # # print(B[0])
        # print(C)
        X = []
        for x in A:
            X += x
        X += B
        X += C
        X = np.asarray(X)
        # print(len(A), len(A[0]))
        # print(len(B))
        # print(len(C))
        # print(X)
        # print("shape of X ",X.shape)

        label = self.data_frame["Solution"].iloc[idx]
        label = np.asarray(eval(label))

        sample = {'X': X, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        # return sample
        return (sample['X'], sample['label'])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, label = sample['X'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {'X': torch.from_numpy(X).float(),
                'label': torch.from_numpy(label).float()}