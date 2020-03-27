import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms

N = 94
P_MIN = 1200
P_MAX = 7000
HOUSE_MAX = 9000
V_MAX = 40000
DT= 0.5

class EDF_data(Dataset):

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
        
        #getting mean and variance of data
        num_max_analyse = 5000
        num_data = self.data_frame.shape[0]
        house_cons_list = []
        for idx in range(num_data):
            # hc = data_frame["opt_charging_profile_step1"].iloc[idx]
            hc = self.data_frame["house_cons"].iloc[idx]
            hc = eval(hc)
            house_cons_list.append(hc)
            if idx > num_max_analyse:
                break
        self.house_mean = np.mean(house_cons_list, axis = 0)
        self.house_vari = np.sqrt(np.var(house_cons_list, axis = 0))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):


        vehicule_energy_need = self.data_frame["vehicle_energy_need"].iloc[idx]
        vehicule_energy_need = np.asarray(eval(vehicule_energy_need))/P_MAX

        house_cons = self.data_frame["house_cons"].iloc[idx]
        house_cons = np.asarray(eval(house_cons))/P_MAX

        X = np.hstack((vehicule_energy_need, house_cons))
        label = self.data_frame["opt_charging_profile"].iloc[idx]
        label = np.asarray(eval(label))/ P_MAX

        sample = {'X': X, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        # return sample
        return (sample['X'], sample['label'])

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
        X = []
        for x in A:
            X += x
        X += B
        X += C
        X = np.asarray(X)

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