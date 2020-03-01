import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time

import loss_LP as lossLP
import loss_EDF as lossEDF

MAX_SHOW = 15
DIEZ = "##########"
EQUAL = "==============="


class FullyConnectedRegularized(nn.Module):
    def __init__(self, num_in_param, num_out_var, num_depth=0, num_neur = 100, dropout=False):
        super(FullyConnectedRegularized, self).__init__()

        # self.l2_reg = l2_reg
        self.num_param = num_in_param

        self.layer_list = []
        fcIn = nn.Linear(self.num_param, num_neur)
        fcOut = nn.Linear(num_neur, num_out_var)

        if dropout:
            self.layer_list.append(nn.Dropout(0.2))
        self.layer_list.append(fcIn)
        self.layer_list.append(nn.ReLU())
        if dropout:
            self.layer_list.append(nn.Dropout(0.5))
        for depth in range(num_depth):
            self.layer_list.append(nn.Linear(num_neur, num_neur))
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
        # print("input shape: ", inputs.shape)
        # print("target shape: ", targets.shape)

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
    # print("Tot penalty ", penalty/N)
    return tot_loss/N, correct/N, cost/N, penalty/N


def test(model, loader, f_loss, device, final_test=False, log_manager = None):
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
                # print("Final test, ",i)
                # example_text += lossLP.print_costs(i, outputs,
                #                             targets, inputs, num_const)
                log_manager.write_example(i, outputs, targets, inputs)
    # print("Tot penalty ", penalty/N)
    return tot_loss/N, correct/N, cost/N, penalty/N


