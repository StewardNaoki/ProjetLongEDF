import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time


class FullyConnectedRegularized0(nn.Module):
    def __init__(self, num_param, num_var, l2_reg=0):
        super(FullyConnectedRegularized0, self).__init__()

        self.l2_reg = l2_reg
        self.num_param = num_param

        self.fcIn = nn.Linear(num_param, 100)
        self.fcOut = nn.Linear(100, num_var)

        self.Layers = nn.Sequential(
            self.fcIn,
            nn.ReLU(),
            self.fcOut
        )

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_param), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_param, x.shape[1])
        output = self.Layers(x)
        return output


class FullyConnectedRegularized1(nn.Module):
    def __init__(self, num_param, num_var, l2_reg=0):
        super(FullyConnectedRegularized1, self).__init__()

        self.l2_reg = l2_reg
        self.num_param = num_param

        self.fcIn = nn.Linear(num_param, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fcOut = nn.Linear(100, num_var)

        self.Layers = nn.Sequential(

            nn.Dropout(0.2),
            self.fcIn,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc1,
            nn.ReLU(),
            self.fcOut
        )

    # def penalty(self):
    #     #L2 regularisation
    #     return self.l2_reg * (self.fc1.weight.norm(2) + self.fcOut.weight.norm(2) + self.fcIn.weight.norm(2))

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_param), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_param, x.shape[1])
        output = self.Layers(x)
        return output


class FullyConnectedRegularized2(nn.Module):
    def __init__(self, num_param, num_var, l2_reg=0):
        super(FullyConnectedRegularized2, self).__init__()

        self.l2_reg = l2_reg
        self.num_param = num_param

        self.fcIn = nn.Linear(num_param, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fcOut = nn.Linear(100, num_var)

        self.Layers = nn.Sequential(

            nn.Dropout(0.2),
            self.fcIn,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fcOut
        )

    def penalty(self):
        #L2 regularisation
        return self.l2_reg * (self.fcIn.weight.norm(2) + self.fc1.weight.norm(2) + self.fc2.weight.norm(2) + self.fcOut.weight.norm(2))

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_param), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_param, x.shape[1])
        output = self.Layers(x)
        return output


class FullyConnectedRegularized3(nn.Module):
    def __init__(self, num_param, num_var, l2_reg=0):
        super(FullyConnectedRegularized3, self).__init__()

        self.l2_reg = l2_reg
        self.num_param = num_param

        self.fcIn = nn.Linear(num_param, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fcOut = nn.Linear(100, num_var)

        self.Layers = nn.Sequential(

            nn.Dropout(0.2),
            self.fcIn,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.ReLU(),
            self.fcOut
        )

    def penalty(self):
        #L2 regularisation
        return self.l2_reg * (self.fcIn.weight.norm(2) + self.fc1.weight.norm(2) + self.fc2.weight.norm(2) + self.fc3.weight.norm(2) + self.fcOut.weight.norm(2))

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_param), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_param, x.shape[1])
        output = self.Layers(x)
        return output


def train(model, loader, f_loss, optimizer, device, custom_loss=False):
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
        if(custom_loss):
            loss = f_loss(outputs, targets, inputs)
            # print("loss ", loss)
            tot_loss += inputs.shape[0] * \
                f_loss(outputs, targets, inputs).item()
            c, p = f_loss.get_info()
            cost += c
            penalty += p

        else:
            loss = f_loss(outputs, targets)
            # print("loss ", loss)
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
            c, p = f_loss.get_info(outputs, targets, inputs)
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


def test(model, loader, f_loss, device, final_test=False, custom_loss=False):
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

            if(custom_loss):
                tot_loss += inputs.shape[0] * \
                    f_loss(outputs, targets, inputs).item()
                c, p = f_loss.get_info()
                cost += c
                penalty += p
            else:
                tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()
                c, p = f_loss.get_info(outputs, targets, inputs)
                cost += c
                penalty += p

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs
            correct += (predicted_targets == targets).sum().item()

            if final_test:
                print_costs(outputs, targets, inputs)

    return tot_loss/N, correct/N, cost/N, penalty/N


def print_costs(outputs, targets, inputs):
        """Function that print the cost of a given output and target

        Arguments:
            outputs {torch tensor} -- output of the model
            targets {torch tensor} -- target
            inputs {torch tensor} -- constraints of the problem
        """
        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        # print(output_cost.shape)
        print("targets cost: ", float(target_cost[0]))
        print("outputs cost: ", float(output_cost[0]))
        print("targets: ", targets[0])
        print("outputs: ", outputs[0])
        print("\n\n")


class CustomMSELoss():
    def __init__(self, num_const=8):
        self.num_const = num_const
        self.f_loss = nn.MSELoss()

    def get_info(self, outputs, targets, inputs):

        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        cost = float(torch.mean((output_cost - target_cost)**2))

        a_const = inputs[:, :-self.num_const -
                         num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
        b_const = inputs[:, -self.num_const -
                         num_var:-num_var].view(num_batch, 1, -1)

        print("a_const: ", a_const)
        print("b_const: ", b_const)

        outputs_penalty = (torch.clamp((torch.bmm(outputs.view(
            num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
        target_penalty = (torch.clamp((torch.bmm(targets.view(
            num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)

        penalty = float(torch.mean(
            (outputs_penalty - target_penalty)**2))
        return cost, penalty

    def __call__(self, outputs, targets):
        return self.f_loss(outputs, targets)


class CustomLoss():

    def __init__(self, alpha=0, num_const=8):
        self.alpha = alpha
        self.num_const = num_const
        self.cost_diff = 0.0
        self.const_penalty = 0.0

    def get_info(self):
        # print("cost diff: ", self.cost_diff)
        # print("const penalty: ", self.const_penalty)
        return self.cost_diff, self.const_penalty

    def __call__(self, outputs, targets, inputs):
        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]
        # print(output.view(num_batch, 1 ,self.num_const))
        # print(inputs[:,-self.num_const:].view(num_batch, self.num_const, 1))
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        # print(target_cost)
        # print(output_cost)
        self.cost_diff = float(torch.mean((output_cost - target_cost)**2))

        result = torch.mean((output_cost - target_cost)**2)

        if self.alpha != 0:
            a_const = inputs[:, :-self.num_const -
                             num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
            b_const = inputs[:, -self.num_const -
                             num_var:-num_var].view(num_batch, 1, -1)

            outputs_penalty = (torch.clamp((torch.bmm(outputs.view(
                num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
            target_penalty = (torch.clamp((torch.bmm(targets.view(
                num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)

            self.const_penalty = float(torch.mean(
                (outputs_penalty - target_penalty)**2))

            result += self.alpha * \
                torch.mean((outputs_penalty - target_penalty)**2)

            # negative_penalty = -1*torch.clamp(outputs,max = 0)

            # result += self.alpha *torch.mean(negative_penalty**2)

        return result

        # print("name of parameters ",name)


# class CustomLoss2():
#     def __init__(self, num_const):
#         self.num_const = num_const

#     def __call__(self, outputs, targets, inputs):
#         # print(outputs.shape)
#         # print(targets.shape)
#         # print(inputs.shape)
#         num_batch = outputs.shape[0]
#         num_var = outputs.shape[1]
#         # print(num_batch)
#         # print("print costs")
#         # print(output.view(num_batch, 1 ,self.num_const))
#         # print(inputs[:,-self.num_const:].view(num_batch, self.num_const, 1))
#         a_const = inputs[:, :-self.num_const -
#                          num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
#         b_const = inputs[:, -self.num_const-num_var:-num_var].view(num_batch, 1, -1)
#         # print("print consts")
#         # print(a_const)
#         # print(b_const)
#         outputs_penalty = (torch.clamp((torch.bmm(outputs.view(
#             num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
#         target_penalty = (torch.clamp((torch.bmm(targets.view(
#             num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
#         # print(torch.clamp((torch.bmm(outputs.view(num_batch, 1 ,self.num_var), a_const) - b_const) , min= 0))
#         # print("print reg")
#         # print(outputs_penalty)
#         # print(target_penalty)

#         outputs_cost = torch.bmm(outputs.view(
#             num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
#         targets_cost = torch.bmm(targets.view(
#             num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))

#         # print("costs ",outputs_cost)
#         # print("costs ",targets_cost)

#         # print(outputs_cost.shape)
#         # print(outputs_penalty.shape)
#         outputs_cost += outputs_penalty.view(num_batch, 1, -1)
#         targets_cost += target_penalty.view(num_batch, 1, -1)
#         # print("total ",outputs_cost)
#         # print("total ",targets_cost)

#         return torch.mean((outputs_cost - targets_cost)**2)


# class ConstraintsPenalty():

#     def __init__(self, num_const, alpha):
#         self.num_const = num_const
#         self.alpha = alpha

#     def __call__(self, outputs, targets, inputs):
#         # print(outputs.shape)
#         # print(targets.shape)
#         # print(inputs.shape)
#         num_batch = outputs.shape[0]
#         num_var = outputs.shape[1]

#         a_const = inputs[:, :-self.num_const -
#                          num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
#         b_const = inputs[:, -self.num_const-num_var:-num_var].view(num_batch, 1, -1)


#         outputs_penalty = (torch.clamp((torch.bmm(outputs.view(
#             num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)
#         target_penalty = (torch.clamp((torch.bmm(targets.view(
#             num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)


#         return self.alpha * torch.mean((outputs_penalty - target_penalty)**2)
