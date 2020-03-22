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

        # print(self.data_frame.head())

        # house_pmax = self.data_frame["house_pmax"].iloc[idx]
        # house_pmax = eval(house_pmax)

        # vehicule_pmax = self.data_frame["vehicle_pmax"].iloc[idx]
        # vehicule_pmax = eval(vehicule_pmax)

        vehicule_energy_need = self.data_frame["vehicle_energy_need"].iloc[idx]
        vehicule_energy_need = np.asarray(eval(vehicule_energy_need))

        house_cons = self.data_frame["house_cons"].iloc[idx]
        house_cons = np.asarray(eval(house_cons))/P_MAX
        # house_cons = np.divide(house_cons - self.house_mean, self.house_vari)

        # print("vehicule_energy_need", vehicule_energy_need)
        # print("house_cons", house_cons)

        X = np.hstack((vehicule_energy_need/P_MAX, house_cons))
        # print(X)
        # X = []
        # # X += house_pmax
        # # X += vehicule_pmax
        # # X += [(vehicule_energy_need[0] - N*P_MIN*DT) / (P_MAX - P_MIN)]
        # X += [(vehicule_energy_need[0]) / (P_MAX)]
        # # print(len(house_cons))
        # for x in house_cons:
        #     X.append(x/HOUSE_MAX)

        # X = np.asarray(X)

        # print(X)
        # label = self.data_frame["opt_charging_profile_step1"].iloc[idx]
        label = self.data_frame["opt_charging_profile"].iloc[idx]
        label = np.asarray(eval(label))
        # label = label[53:67]
        # label = (label - P_MIN)/(P_MAX - P_MIN)
        label = (label) / P_MAX

        # print("X shape ", X.shape)
        # print("label shape ",label.shape)

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