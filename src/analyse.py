import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

# num = 5000
# P_MAX = 9000
# csv_file_name = "../DATA/inputNJM{}.csv".format(num)
# data_frame = pd.read_csv(csv_file_name)
# print(data_frame.head())
# print(data_frame.shape[0])
# house_cons_list = []
# mean = []
# for idx in range(num):
#     hc = data_frame["opt_charging_profile"].iloc[idx]
#     # hc = data_frame["house_cons"].iloc[idx]
#     hc = eval(hc)
#     # mean.append(sum(hc)/len(hc))
#     house_cons_list.append(hc)
#     # print(house_cons_list)
#     # if idx == 5:
#     #     break
# house_cons_list = np.asarray(house_cons_list)
# house_mean = np.mean(house_cons_list, axis = 0)
# # house_vari = np.sqrt(np.var(house_cons_list, axis = 0))
# # house_cons_list =np.divide(house_cons_list - house_mean, house_vari)
# # print(house_mean.shape)
# # print(house_cons_list.shape)

# print(house_mean)

# fig = plt.figure()
# plt.plot(house_mean)
# # plt.plot(house_cons_list[0])
# # plt.legend()
# plt.show()

# # house_mean = house_mean[house_mean != 0]
# # print(house_mean)
# for k in range(len(house_mean)):
#     if house_mean[k] != 0:
#         print(k)
#         print(house_mean[k])

# a = np.array([[1,2,3],[4,5,6]])
# mean = np.array([0.5,1,0.2])
# print(a-mean)
# var = np.array([0.25,0.25,0.5])
# a = np.array()
# a = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# a.view(a.shape[0],-1)


# a = torch.tensor([[[1,2,3,4],[15,6,7,8],[30,4,5,6]], [[1,2,3,4],[50,6,7,8],[30,4,5,6]]])

# need = torch.tensor([10,12])
# # begin = a[:, :, 0].view(a.shape[0], -1,a.shape[1])
# begin = a[:, :, 0]
# width = a[:, :, 1]
# power = a[:, :, 2]
# # certainty = a[:, :, 3].view(a.shape[0], -1,a.shape[1])
# certainty = a[:, :, 3]
relu = nn.ReLU()
# print(begin.shape)
# print(begin)
# print(width)
# print(power)
# print(relu(power-1))
# print(certainty)
# print(power * certainty)
# print((power * certainty*width*(1/2)).sum(dim = 1))
# print(power * certainty*width*(1/2))
# print(((need - (power * certainty*width*(1/2)).sum(dim = 1))**2).mean())



# pve = torch.arange(94)

# print(pve)
# c = torch.max(pve )
# maxim = torch.zeros(a.shape[0])
# for i in range (a.shape[0]):
#     pve_copy = pve
#     line_zero = torch.zeros(94)
#     for j in range(3):
#         pve_copy[begin[i,j]: begin[i,j] + width[i,j]] += power[i,j]
#         line_zero[begin[i,j]: begin[i,j] + width[i,j]] += power[i,j]
#     maxim[i] = torch.max(pve_copy)
#     plt.plot(pve_copy)
#     plt.plot(line_zero)
#     plt.show()
#     # x = [pve[begin[i,j]: begin[i,j] + width[i,j]]+power[i,j] for j in range (3)]
# # pve2 = torch.tensor([[for intervall in begin] for batch in range(a.shape[0])])

# print(maxim)
# print(torch.mean(maxim))

b = torch.tensor([[2,0,0,1,2,1,3,4],[3,1,1,0,0,3,3,3]])
# b[:,:,0] = 4
pve = b[:,1:]
need = b[:,0]
# need = b[:,0]
print(pve, need)
pve_correct = 2*(pve==0).float()
print(pve_correct)
# print(torch.bmm(need,pve_correct))
# print((need*pve_correct).view(pve.shape[0], pve.shape[2]))
# # c.asfloat()
# print(b)
# print(pve_correct)
# print(b + b_correct)
c= pve+pve_correct
# print((relu((c - (1/4))).sum()))
# print((relu((c - (1/4)))))
print((relu(((2) - c))).sum(dim = 1).mean())
print((relu(((2) - c))))
print((relu((c - (1/4)))))
print((need - (0.5 * pve).sum(dim=1)))
penalty_need = ((need - (0.5 * pve).sum(dim=1))**2).sum()
print(penalty_need)



























