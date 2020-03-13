import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

num = 1000
P_MAX = 7000
csv_file_name = "../DATA/inputNJM{}.csv".format(num)
data_frame = pd.read_csv(csv_file_name)
print(data_frame.head())

house_cons_list = []
mean = []
for idx in range(num):
    hc = data_frame["opt_charging_profile_step1"].iloc[idx]
    hc = eval(hc)
    mean.append(sum(hc)/len(hc))
    house_cons_list.append(hc)
    # print(house_cons_list)
    if idx == 5:
        break
house_cons_list = np.asarray(house_cons_list)
house_mean = np.mean(house_cons_list, axis = 0) / P_MAX
print(house_mean.shape)
# print(house_cons_list.shape)


fig = plt.figure()
plt.plot(house_mean)
# plt.legend()
plt.show()


# a = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# a.view(a.shape[0],-1)