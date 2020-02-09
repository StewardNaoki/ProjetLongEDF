import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time

MAX_SHOW = 15
DIEZ = "##########"
EQUAL = "==============="


class FullyConnectedRegularized(nn.Module):
    def __init__(self, num_param, num_var, l2_reg=0, num_depth=0, dropout=False):
        super(FullyConnectedRegularized, self).__init__()

        self.l2_reg = l2_reg
        self.num_param = num_param

        self.layer_list = []
        fcIn = nn.Linear(num_param, 100)
        fcOut = nn.Linear(100, num_var)

        if dropout:
            self.layer_list.append(nn.Dropout(0.2))
        self.layer_list.append(fcIn)
        self.layer_list.append(nn.ReLU())
        if dropout:
            self.layer_list.append(nn.Dropout(0.5))
        for depth in range(num_depth):
            self.layer_list.append(nn.Linear(100, 100))
            self.layer_list.append(nn.ReLU())

        self.layer_list.append(fcOut)

        self.Layers = nn.Sequential(*self.layer_list)

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_param), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_param, x.shape[1])
        output = self.Layers(x)
        return output


def train(model, loader, f_loss, optimizer, device):
    """Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Keyword Arguments:
        custom_loss boolean -- test if custom loss is used (default: {False})

    Returns:

    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0
    cost, penalty = 0.0, 0.0
    for i, (inputs, targets) in enumerate(loader):
        # pbar.update(1)
        # pbar.set_description("Training step {}".format(i))

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)

        #get loss

        loss = f_loss(outputs, targets, inputs)
        # print("loss ", loss)
        tot_loss += inputs.shape[0] * \
            f_loss(outputs, targets, inputs).item()
        c, p = f_loss.get_info()
        cost += c
        penalty += p

        N += inputs.shape[0]

        #test if loss has a defined gradient
        assert(loss.requires_grad), "No gradient for loss"

        # print("Output: ", outputs)
        predicted_targets = outputs

        correct += (predicted_targets == targets).sum().item()

        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        # model.penalty().backward()
        optimizer.step()

    return tot_loss/N, correct/N, cost/N, penalty/N


def test(model, loader, f_loss, device, final_test=False, num_const=0):
    """Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation


    Keyword Arguments:
        final_test boolean -- test if this is the final test: the function will show some result (default: {False})
        custom_loss boolean -- test if custom loss is used (default: {False})

    Returns:
        A tuple with the mean loss and mean accuracy
    """

    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        cost, penalty = 0.0, 0.0
        example_text = ""

        for i, (inputs, targets) in enumerate(loader):
            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples

            tot_loss += inputs.shape[0] * \
                f_loss(outputs, targets, inputs).item()
            c, p = f_loss.get_info()
            cost += c
            penalty += p


            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs
            correct += (predicted_targets == targets).sum().item()

            if final_test and i < MAX_SHOW:
                example_text += print_costs(i, outputs,
                                            targets, inputs, num_const)

    return tot_loss/N, correct/N, cost/N, penalty/N, example_text


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
        A = a_const.transpose(2, 1)[0]
        B = b_const[0]
        C = inputs[:, -num_var:][0]

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


""".format(num, A, B, C, float(target_cost[0]), float(output_cost[0]), output_penalty[0], targets[0], outputs[0])

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

            result += res

        return result



def compute_penalty(outputs, inputs, alpha, num_batch, num_const, num_var):
    a_const = inputs[:, :-num_const -
                     num_var].view(num_batch, num_const, num_var).transpose(2, 1)
    b_const = inputs[:, -num_const -
                     num_var:-num_var].view(num_batch, 1, -1)

    output_penalty = (torch.clamp((torch.bmm(outputs.view(
        num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
    return abs(float(torch.mean((output_penalty)))),  torch.mean(output_penalty)*(alpha)
