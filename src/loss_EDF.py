import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time

MAX_SHOW = 15
DT = 0.5
HOUSE_MAX = 9000
P_MAX = 7000
P_MIN = 1200
DIEZ = "##########"
EQUAL = "==============="


def print_costs(num, outputs, targets, inputs, num_const):
        """Function that print the cost of a given output and target

        Arguments:
            outputs {torch tensor} -- output of the model
            targets {torch tensor} -- target
            inputs {torch tensor} -- constraints of the problem
        """
        example_text = ""
        num_batch = outputs.shape[0]
        house_cons = inputs[:, 3:]
        # loss = torch.mean(((torch.max(house_cons + outputs, 1)[0]) - (torch.max(house_cons + targets, 1)[0]))**2)
        output_cost = torch.max(house_cons + outputs, 1)[0]
        target_cost = torch.max(house_cons + targets, 1)[0]

        example_text += """Example {}
===============

Problem
===============


Costs
===============
targets cost: {}


output cost: {}


Penalty
===============
output penalty: {}


""".format(num, float(target_cost[0]), float(output_cost[0]), output_penalty[0])

        print("example_text: ", example_text)

        return example_text


class CustomMSELoss():

    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta
        self.f_loss = nn.MSELoss()
        self.cost = 0.0
        self.penalty = 0.0

    def get_info(self):
        # print("cost diff: ", self.cost_diff)
        # print("const penalty: ", self.const_penalty)
        return self.cost, self.penalty

    def __call__(self, outputs, targets, inputs):
        loss = self.f_loss(outputs, targets)
        # print(outputs, targets)
        self.cost = float(loss)
        self.penalty, res = compute_penalty(outputs, inputs, self.alpha, self.beta)
        # print("loss ", loss)
        # print("outputs ", outputs[0])
        # print("targets ", targets[0])
        # print("penalty ", res)
        return loss + res
        # return loss


class PureCostLoss():

    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta
        # self.f_loss = nn.MSELoss()
        self.cost = 0.0
        self.penalty = 0.0

    def get_info(self):
        # print("cost diff: ", self.cost_diff)
        # print("const penalty: ", self.const_penalty)
        return self.cost, self.penalty

    def __call__(self, outputs, targets, inputs):
        # loss = self.f_loss(outputs, targets) +
        house_cons = inputs[:, 1:]
        # compute_penalty(outputs, inputs, self.alpha, self.beta)
        loss = torch.mean(torch.max(house_cons + outputs, 1)[0])
        self.cost = float(loss)
        self.penalty, res = compute_penalty(
            outputs, inputs, self.alpha, self.beta)
        return loss + res


class GuidedCostLoss():

    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        self.beta = beta
        # self.f_loss = nn.MSELoss()
        self.cost = 0.0

    def get_info(self):
        # print("cost diff: ", self.cost_diff)
        # print("const penalty: ", self.const_penalty)
        return self.cost, self.penalty
        # self.penalty = 0.0

    def __call__(self, outputs, targets, inputs):
        # loss = self.f_loss(outputs, targets) +
        house_cons = inputs[:, 3:]
        # compute_penalty(outputs, inputs, self.alpha, self.beta)
        # output_cost = (torch.max(house_cons + outputs, 1)[0]).sum()
        # target_cost = (torch.max(house_cons + targets, 1)[0]).sum()
        loss = torch.mean(((torch.max(house_cons + outputs, 1)
                            [0]) - (torch.max(house_cons + targets, 1)[0]))**2)
        self.cost = float(loss)
        self.penalty, res = compute_penalty(
            outputs, inputs, self.alpha, self.beta)
        return loss + res
    

# class IntervalCostLoss():

#     def __init__(self, alpha=0, beta=0):
#         self.alpha = alpha
#         self.beta = beta
#         # self.f_loss = nn.MSELoss()
#         self.cost = 0.0
#         self.penalty = 0.0

#     def get_info(self):
#         # print("cost diff: ", self.cost_diff)
#         # print("const penalty: ", self.const_penalty)
#         return self.cost, self.penalty

#     def __call__(self, outputs, targets, inputs):
#         # loss = self.f_loss(outputs, targets) +
#         begin = (outputs[:, :, 0]*94).int()
#         width = (outputs[:, :, 1]*94).int()
#         power = outputs[:, :, 2]
#         certainty = outputs[:, :, 3]
#         print("begin ", begin)
#         print("width ", width)
#         print("power ", power)
#         print("certainty ", certainty)
#         relative_pow = power * certainty
#         # compute_penalty(outputs, inputs, self.alpha, self.beta)
#         maxim = torch.zeros(outputs.shape[0])
#         for i in range (outputs.shape[0]):
#             pve = outputs
#             line_zero = torch.zeros(94)
#             for j in range(3):
#                 pve[begin[i,j]: begin[i,j] + width[i,j]] += relative_pow[i,j]
#                 line_zero[begin[i,j]: begin[i,j] + width[i,j]] += relative_pow[i,j]
#             maxim[i] = torch.max(pve)
#             # plt.plot(pve)
#             # plt.plot(line_zero)
#             # plt.show()
#         loss = torch.mean(maxim)
#         # loss = torch.mean(torch.max(house_cons + outputs, 1)[0])
#         self.cost = float(loss)
#         self.penalty, res = compute_penalty_interval(
#             outputs, inputs, self.alpha, self.beta)
#         return loss + res


def compute_penalty(outputs, inputs, alpha, beta):
    pve = outputs
    need = inputs[:, 1]
    relu = nn.ReLU()
    power_correct = pve + (P_MIN/P_MAX) * (pve == 0).float()
    penalty_max = (relu((power_correct - (P_MAX/P_MAX)))).sum(dim = 1).mean()
    penalty_min = (relu(((P_MIN/P_MAX) - pve))).sum(dim = 1).mean()
    # print("pve min: ", (relu(( (P_MIN/P_MAX)- pve.t()).t())[0]))
    # print("pve max: ", (relu((pve.t() - 1).t())[0]))
    # penalty_min = (relu(-pve)).sum()
    penalty_need = ((need - (DT * pve).sum(dim=1))**2).sum()
    penalty = alpha*(penalty_max + penalty_min) + beta*(penalty_need)
    # print("Outputs: ", pve )
    # print("penalty max: ", penalty_max )
    # print("penalty_min: ", penalty_min )
    # print("penalty_need: ", penalty_need )
    # print("pmax: ",pmax)
    # print("need: ",need)
    # print("penalty: ", penalty )

    return float(penalty), penalty


# def compute_penalty_interval(outputs, inputs, alpha, beta):
#     pve = outputs
#     begin = outputs[:, :, 0]
#     width = outputs[:, :, 1]
#     power = outputs[:, :, 2]
#     certainty = outputs[:, :, 3]
#     relative_pow = power * certainty
#     power_correct = relative_pow + (P_MIN/P_MAX) * (relative_pow == 0).float()
#     # pmax = inputs[:, 1]
#     need = inputs[:, 1]
#     relu = nn.ReLU()
#     penalty_max = (relu((power - (P_MAX/P_MAX)))).sum(dim = 1).mean()
#     penalty_min = (relu(((P_MIN/P_MAX) - power_correct))).sum(dim = 1).mean()
#     # print("pve min: ", (relu(( (P_MIN/P_MAX)- pve.t()).t())[0]))
#     # print("pve max: ", (relu((pve.t() - 1).t())[0]))
#     # penalty_min = (relu(-pve)).sum()
    
#     penalty_need = ((((need - (power_correct * width * DT).sum(dim=1))**2).mean()))
#     penalty = alpha*(penalty_max + penalty_min) + beta*(penalty_need)
#     # print("Outputs: ", pve )
#     # print("penalty max: ", penalty_max )
#     # print("penalty_min: ", penalty_min )
#     # print("penalty_need: ", penalty_need )
#     # print("pmax: ",pmax)
#     # print("need: ",need)
#     # print("penalty: ", penalty )

#     return float(penalty), penalty
