import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from sklearn.svm import SVC

N = 94
P_MIN = 1200
P_MAX = 7000
HOUSE_MAX = 9000
V_MAX = 40000
DT= 0.5
num_data = 5000
csv_file_name = "inputNJM{}.csv".format(num_data)

data_frame = pd.read_csv(csv_file_name)

clf = SVC(gamma='auto')

with tqdm(total=num_data) as pbar:
    for idx in range(num_data):
        pbar.update(1)
        pbar.set_description("Epoch {}".format(idx))
        house_cons = data_frame["house_cons"].iloc[idx]
        house_cons = np.asarray(eval(house_cons))/P_MAX
        vehicule_energy_need = data_frame["vehicle_energy_need"].iloc[idx]
        vehicule_energy_need = np.asarray(eval(vehicule_energy_need))
        X = np.hstack((vehicule_energy_need/P_MAX, house_cons))
        Y = data_frame["opt_charging_profile"].iloc[idx]
        Y = np.asarray(eval(Y))
    