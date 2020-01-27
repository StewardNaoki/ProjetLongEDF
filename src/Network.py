import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time


class FullyConnectedRegularized(nn.Module):
    def __init__(self, num_param, num_var, l2_reg):
        super(FullyConnectedRegularized, self).__init__()

        self.l2_reg = l2_reg
        self.num_param = num_param

        # fully connected layer, output 10 classes
        self.fcIn = nn.Linear(num_param, 100)
        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(100, 100)
        # fully connected layer, output 10 classes
        # self.fc2 = nn.Linear(100, 100)
        # # fully connected layer, output 10 classes
        # self.fc3 = nn.Linear(100, 100)
        # fully connected layer, output 10 classes
        self.fcOut = nn.Linear(100, num_var)

        self.Layers = nn.Sequential(

            nn.Dropout(0.2),
            self.fcIn,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc1,
            nn.ReLU(),
            # self.fc2,
            # nn.ReLU(),
            # self.fc3,
            # nn.ReLU(),
            self.fcOut
        )

    def penalty(self):
        # return self.l2_reg * (self.fc1.weight.norm(2) + self.fc2.weight.norm(2) + self.fc3.weight.norm(2) + self.fcFinal.weight.norm(2))
        return self.l2_reg * (self.fc1.weight.norm(2)  + self.fcFinal.weight.norm(2) + self.fcIn.weight.norm(2))

    def forward(self, x):
        assert (x.shape[1] == self.num_param),"Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(self.num_param, x.shape[1])
        # x = x.view(x.size(0), -1)
        # print("num_param =", self.num_param)
        # print("true num_param =", x.shape)
        output = self.Layers(x)
        # return output, x    # return x for visualization
        return output


def train(model, loader, f_loss, optimizer, device, custom_loss = False):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0
    for i, (inputs, targets) in enumerate(loader):
        # pbar.update(1)
        # pbar.set_description("Training step {}".format(i))
        # print("****", inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        # print("***",inputs.shape)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)

        if(custom_loss):
            loss = f_loss(outputs, targets, inputs)
            # print("loss ", loss)
            tot_loss += inputs.shape[0] * f_loss(outputs, targets, inputs).item()
        else:

            loss = f_loss(outputs, targets)
            # print("loss ", loss)
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

        N += inputs.shape[0]

        assert(loss.requires_grad), "No gradient for loss"

        # print("Output: ", outputs)
        predicted_targets = outputs

        correct += (predicted_targets == targets).sum().item()

        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        # model.penalty().backward()
        optimizer.step()
    return tot_loss/N, correct/N


def test(model, loader, f_loss, device, final_test=False, custom_loss = False):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation

    Returns :

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
                tot_loss += inputs.shape[0] * f_loss(outputs, targets, inputs).item()
            else:
                tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs
            correct += (predicted_targets == targets).sum().item()

            if final_test:
                print("targets:\n", targets[0])
                print("predicted targets:\n", outputs[0])

    return tot_loss/N, correct/N


# def custom_locc(output, target, input):
#     loss = torch.mean((output - target)**2)
#     return loss

class CustomLoss():
    def __init__(self, num_cont):
        self.num_const =num_cont

    def __call__(self, output, target, inputs):
        num_batch = output.shape[0]
        # print(output.view(num_batch, 1 ,self.num_const))
        # print(inputs[:,-self.num_const:].view(num_batch, self.num_const, 1))
        output_cost = torch.bmm(output.view(num_batch, 1 ,self.num_const), inputs[:,-self.num_const:].view(num_batch, self.num_const, 1))
        target_cost = torch.bmm(target.view(num_batch, 1 ,self.num_const), inputs[:,-self.num_const:].view(num_batch, self.num_const, 1))
        # print(target_cost)
        # print(output_cost)
        return torch.mean((output_cost - target_cost)**2)
