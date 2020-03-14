import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

num = 1000
P_MAX = 9000
csv_file_name = "../DATA/inputNJM{}.csv".format(num)
data_frame = pd.read_csv(csv_file_name)
print(data_frame.head())
print(data_frame.shape[0])
house_cons_list = []
mean = []
for idx in range(num):
    hc = data_frame["opt_charging_profile_step1"].iloc[idx]
    # hc = data_frame["house_cons"].iloc[idx]
    hc = eval(hc)
    mean.append(sum(hc)/len(hc))
    house_cons_list.append(hc)
    # print(house_cons_list)
    # if idx == 5:
    #     break
house_cons_list = np.asarray(house_cons_list)
house_mean = np.mean(house_cons_list, axis = 0)
house_vari = np.sqrt(np.var(house_cons_list, axis = 0))
house_cons_list =np.divide(house_cons_list - house_mean, house_vari)
# print(house_mean.shape)
# print(house_cons_list.shape)


fig = plt.figure()
# plt.plot(house_mean)
plt.plot(house_cons_list[0])
# plt.legend()
plt.show()

# a = np.array([[1,2,3],[4,5,6]])
# mean = np.array([0.5,1,0.2])
# print(a-mean)
# var = np.array([0.25,0.25,0.5])
# a = np.array()
# a = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# a.view(a.shape[0],-1)