import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

num = 5000
P_MAX = 9000
csv_file_name = "../DATA/inputNJM{}.csv".format(num)
data_frame = pd.read_csv(csv_file_name)
print(data_frame.head())
print(data_frame.shape[0])
house_cons_list = []
house_cons_list1 = []
mean = []
for idx in range(num):
    hc1 = data_frame["house_cons"].iloc[idx]
    hc = data_frame["opt_charging_profile"].iloc[idx]
    # hc = data_frame["house_cons"].iloc[idx]
    hc = eval(hc)
    hc1 = eval(hc1)
    # mean.append(sum(hc)/len(hc))
    house_cons_list.append(hc)
    house_cons_list1.append(hc1)
    # print(hc1)
    # print(house_cons_list)
    # if idx == 5:
    #     break
house_cons_list = np.asarray(house_cons_list)/P_MAX
house_cons_list1 = np.asarray(house_cons_list1)/P_MAX
house_mean = np.mean(house_cons_list, axis = 0)
house_mean1 = np.mean(house_cons_list1, axis = 0)
house_vari = np.sqrt(np.var(house_cons_list, axis = 0))
# house_cons_list =np.divide(house_cons_list - house_mean, house_vari)
# print(house_mean.shape)
# print(house_cons_list.shape)

print(house_mean)

fig = plt.figure()
plt.plot(house_mean)
plt.plot(house_mean1)
plt.show()
fig = plt.figure()
plt.plot(house_mean1)
for k in range(10):
    plt.plot(house_cons_list[k])
plt.show()
fig = plt.figure()
plt.plot(house_vari)
plt.show()


























