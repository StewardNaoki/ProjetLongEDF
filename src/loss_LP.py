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
        num_var = outputs.shape[1]
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))

        a_const = inputs[:, :-num_const -
                         num_var].view(num_batch, num_const, num_var).transpose(2, 1)
        b_const = inputs[:, -num_const -
                         num_var:-num_var].view(num_batch, 1, -1)

        output_penalty = (torch.clamp((torch.bmm(outputs.view(
            num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)

        negative_penalty = nn.ReLU()(-outputs).sum(dim = 1).view(num_batch,1)

        # print(negative_penalty)
        # print(output_penalty)
        output_penalty += negative_penalty

        A = (a_const.transpose(2, 1)[0]).tolist()
        B = (b_const[0]).tolist()
        C = (inputs[:, -num_var:][0]).tolist()

        # print(inputs)
        # print(inputs.shape)

        # print(output_cost.shape)
        # print("targets cost: ", -1*float(target_cost[0]))# ne pas oublier les -1
        # print("outputs cost: ", -1*float(output_cost[0]))
        # print("targets: ", targets[0])
        # print("outputs: ", outputs[0])
        # print("\n\n")

        example_text += """Example {}
===============

Problem
===============
A: {}


B: {}


C: {}

Costs
===============
targets cost: {}


output cost: {}


Penalty
===============
output penalty: {}


Vector
===============
targets vector: {}


output vector: {}


""".format(num, A, B, C, float(target_cost[0]), float(output_cost[0]), output_penalty[0].tolist(), targets[0].tolist(), outputs[0].tolist())

        print("example_text: ", example_text)

        return example_text


class CustomMSELoss():
    def __init__(self,alpha ,num_const=8):
        self.alpha = alpha
        self.num_const = num_const
        self.f_loss = nn.MSELoss()
        self.cost = 0.0
        self.penalty = 0.0


    def get_info(self):

        return self.cost, self.penalty

    def print_info(self):
        print("Cost: ", self.cost)
        print("Penalty: ", self.penalty)

    def __call__(self, outputs, targets, inputs):

        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        self.cost = abs(float(torch.mean((output_cost - target_cost))))

        # a_const = inputs[:, :-self.num_const -
        #                  num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
        # b_const = inputs[:, -self.num_const -
        #                  num_var:-num_var].view(num_batch, 1, -1)

        # output_penalty = (torch.clamp((torch.bmm(outputs.view(
        #     num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
        res = 0
        if self.alpha != 0:
            self.penalty, res = compute_penalty(
                outputs, inputs, self.alpha, num_batch, self.num_const, num_var)
            # print("Penalty ", self.penalty)


        return self.f_loss(outputs, targets) + res


class PureCostLoss():

    def __init__(self, alpha=0, num_const=8):
        self.alpha = alpha
        self.num_const = num_const
        self.cost = 0.0
        self.penalty = 0.0

    def get_info(self):
        # print("cost diff: ", self.cost_diff)
        # print("const penalty: ", self.const_penalty)
        return self.cost, self.penalty

    def print_info(self):
        print("Cost: ", self.cost)
        print("Penalty: ", self.penalty)

    def __call__(self, outputs, targets, inputs):
        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]
        # print(output.view(num_batch, 1 ,self.num_const))
        # print(inputs[:,-self.num_const:].view(num_batch, self.num_const, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        self.cost = abs(float(torch.mean((output_cost - target_cost))))

        result = -1*torch.mean(output_cost)  # ne pas oublier le -1
        # result = torch.mean((output_cost - target_cost)**2)

        if self.alpha != 0:
            # a_const = inputs[:, :-self.num_const -
            #                  num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
            # b_const = inputs[:, -self.num_const -
            #                  num_var:-num_var].view(num_batch, 1, -1)

            # output_penalty = (torch.clamp((torch.bmm(outputs.view(
            #     num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
            self.penalty, res = compute_penalty(
                outputs, inputs, self.alpha, num_batch, self.num_const, num_var)
            # print("Penalty ", self.penalty)


            result += res

        return result

        # print("name of parameters ",name)


class GuidedCostLoss():

    def __init__(self, alpha=0, num_const=8):
        self.alpha = alpha
        self.num_const = num_const
        self.cost = 0.0
        self.penalty = 0.0

    def get_info(self):
        # print("cost diff: ", self.cost_diff)
        # print("const penalty: ", self.const_penalty)
        return self.cost, self.penalty

    def print_info(self):
        print("Cost: ", self.cost)
        print("Penalty: ", self.penalty)

    def __call__(self, outputs, targets, inputs):
        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]

        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        self.cost = abs(float(torch.mean((output_cost - target_cost))))

        result = torch.mean((output_cost - target_cost)**2)

        if self.alpha != 0:
            self.penalty, res = compute_penalty(
                outputs, inputs, self.alpha, num_batch, self.num_const, num_var)
            # print("Penalty ", self.penalty)

            result += res

        return result



def compute_penalty(outputs, inputs, alpha, num_batch, num_const, num_var):
    a_const = inputs[:, :-num_const -
                     num_var].view(num_batch, num_const, num_var).transpose(2, 1)
    b_const = inputs[:, -num_const -
                     num_var:-num_var].view(num_batch, 1, -1)

    output_penalty = (torch.clamp((torch.bmm(outputs.view(
        num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
    # print("input shape ", outputs.shape)
    negative_penalty = nn.ReLU()(-outputs).sum(dim = 1).view(num_batch,1)
    # print("negative_penalty ", negative_penalty)
    # print("output penalty ", output_penalty)
    # print("output penalty ", output_penalty + negative_penalty)
    return abs(float(torch.mean(output_penalty + negative_penalty))),  torch.mean(output_penalty + negative_penalty)*(alpha)
