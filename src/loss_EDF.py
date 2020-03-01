import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time

MAX_SHOW = 15
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
        self.cost = float(loss)
        self.penalty, res = compute_penalty(
            outputs, inputs, self.alpha, self.beta)
        return loss + res


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
        house_cons = inputs[:, 3:]
        # compute_penalty(outputs, inputs, self.alpha, self.beta)
        loss = torch.mean(torch.max(house_cons + outputs, 1)[0])
        self.cost = float(loss)
        self.penalty, res = compute_penalty(
            outputs, inputs, self.alpha, self.beta)
        return loss + res


class GuidedCostLoss():

    def __init__(self, alpha=0, beta=0):
        self.alpha = alpha
        if beta == 0:
            self.beta = alpha
        else:
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


def compute_penalty(outputs, inputs, alpha, beta):
    pve = outputs
    pmax = inputs[:, 1]
    need = inputs[:, 2]
    delta_t = 10
    relu = nn.ReLU()
    p1 = (relu((pve.t() - pmax).t())).sum() + (relu(-pve)).sum()
    p2 = ((need - (delta_t * pve).sum(dim=1))**2).sum()
    penalty = alpha*p1 + beta*p2

    return float(penalty), penalty
