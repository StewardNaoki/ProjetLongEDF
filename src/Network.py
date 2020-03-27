import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np

import loss_LP as lossLP
import loss_EDF as lossEDF

MAX_SHOW = 15
DIEZ = "##########"
EQUAL = "==============="


class CNNSimple(nn.Module):
    def __init__(self, num_in_var = 94, num_out_var = 94, num_depth=0, num_neur = 128, dropout=False):
        super(CNN, self).__init__()

        self.num_in_var = num_in_var
        # self.l2_reg = l2_reg
        in_lenght = num_in_var
        in_channel = 1
        min_in_fc = 4
        max_pool_stride = 2
        max_pool_kernel_size = 3
        cnn_stride = 1
        cnn_kernel_size = 3
        num_channel = 64
        self.cnn_layer_list = []
        self.fc_layer_list = []

        # self.bn1 = nn.BatchNorm2d(32)


        self.cnn_layer_list.append(nn.Conv1d(in_channels=1, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
        self.cnn_layer_list.append(nn.ReLU())
        # self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
        for depth in range(num_depth):
            self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
            self.cnn_layer_list.append(nn.ReLU())
            # self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))

        self.CNN_Layers = nn.Sequential(*self.cnn_layer_list)

    def forward(self, x):
        return fc_output

class CNN2(nn.Module):
    def __init__(self, num_in_var = 94, num_depth=0, num_neur = 128, dropout=False):
        super(CNN2, self).__init__()

        self.num_in_var = num_in_var
        # self.l2_reg = l2_reg
        in_lenght = num_in_var
        in_channel = 1 #Number of channels
        min_in_fc = 4 #Minimal number of channels
        max_pool_stride = 2
        max_pool_kernel_size = 3
        cnn_stride = 1
        cnn_kernel_size = 3
        num_channel = 32 #Number of chennel in each layer
        num_out_var = 94 #size of output vector
        self.cnn_layer_list = [] #List of operation in CNN
        self.fc_layer_list = [] #List of operation in fully connected
        # self.sigmoid = nn.Sigmoid()

        # self.bn1 = nn.BatchNorm2d(32)

        self.cnn_layer_list.append(nn.Conv1d(in_channels=1, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
        in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
        self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
        self.cnn_layer_list.append(nn.ReLU())
        if dropout:
            self.cnn_layer_list.append(nn.Dropout(0.2))


        self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
        in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
        self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
        self.cnn_layer_list.append(nn.ReLU())
        if dropout:
            self.cnn_layer_list.append(nn.Dropout(0.5))


        self.cnn_layer_list.append(nn.MaxPool1d(kernel_size = max_pool_kernel_size, stride=max_pool_stride))
        in_lenght = np.floor(((in_lenght-(max_pool_kernel_size - 1) -1) / max_pool_stride)+1)

        for depth in range(num_depth):

            self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
            in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
            self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
            self.cnn_layer_list.append(nn.ReLU())
            if dropout:
                self.cnn_layer_list.append(nn.Dropout(0.5))


            self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
            in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
            self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
            self.cnn_layer_list.append(nn.ReLU())
            if dropout:
                self.cnn_layer_list.append(nn.Dropout(0.5))



            self.cnn_layer_list.append(nn.MaxPool1d(kernel_size = max_pool_kernel_size, stride=max_pool_stride))
            print(in_lenght)
            in_lenght = np.floor(((in_lenght-(max_pool_kernel_size - 1)-1) / max_pool_stride)+1)
            print(in_lenght)

            assert(in_lenght > min_in_fc), "too deep: depth = {}, in_lenght = {}".format(num_depth, in_lenght)

        num_in_fc = int((num_channel*in_lenght) + 1)
        print("num_in_fc ", num_in_fc)
        fcIn = nn.Linear(num_in_fc, num_neur)
        fcOut = nn.Linear(num_neur, num_out_var)

        self.fc_layer_list.append(fcIn)
        self.fc_layer_list.append(fcOut)
        self.fc_layer_list.append(nn.Sigmoid())

        self.CNN_Layers = nn.Sequential(*self.cnn_layer_list)
        self.FC_Layers = nn.Sequential(*self.fc_layer_list)

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_in_var + 1), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_in_var, x.shape[1])
        house_cons = x[:,1:].view(x.shape[0],-1,x.shape[1]-1)
        need = x[:,0].view(-1, x.shape[0]).t()

        cnn_output = self.CNN_Layers(house_cons)
        cnn_output = cnn_output.view(x.shape[0], -1)
        fc_input = torch.cat([need, cnn_output], dim = 1)
        fc_output = self.FC_Layers(fc_input)

        return fc_output

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

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)


        ###Print test

        # print("Weights\n")
        # print(model.CNN_Layers[0].weight)
        # print(model.FC_Layers[0].weight)
        # print("biases\n")
        # print(model.CNN_Layers[0].bias)
        # print(model.FC_Layers[0].bias)

        # if i ==0 :
        #     before_zero = 53*[0]
        #     after_zero = 27*[0]
        #     # print("inputs ",inputs)
        #     output_pve =before_zero + outputs[:1, :].tolist()[0] + after_zero
        #     target_pve = before_zero +targets[:1, :].tolist()[0] + after_zero
        #     house_cons = inputs[:1, 1:].tolist()[0]
        #     fig = plt.figure()
        #     plt.plot(output_pve[:], label="output_pve")
        #     plt.plot(target_pve[:], label="target_pve")
        #     plt.plot(house_cons[:], label="house_cons")
        #     plt.title('Courbe{}'.format(i))
        #     plt.legend()
        #     plt.show()


        #get loss

        # loss = f_loss(outputs, targets, inputs) * 100
        loss = f_loss(outputs, targets, inputs)
        # print("training loss ", loss)
        tot_loss += inputs.shape[0] * \
            f_loss(outputs, targets, inputs).item()
        cost += c
        penalty += p

        N += inputs.shape[0]

        #test if loss has a defined gradient
        assert(loss.requires_grad), "No gradient for loss"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

            loss = f_loss(outputs, targets, inputs)
            # print("testing loss ", loss)

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
            # correct += (predicted_targets == targets).sum().item()

            if final_test and i < MAX_SHOW:
                # print("Final test, ",i)
                # example_text += lossLP.print_costs(i, outputs,
                #                             targets, inputs, num_const)
                log_manager.write_example(i, outputs, targets, inputs)
    # print("Tot penalty ", penalty/N)
    return tot_loss/N, correct/N, cost/N, penalty/N


