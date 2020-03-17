class FullyConnectedRegularized(nn.Module):
    def __init__(self, num_in_var = 95, num_out_var = 94, num_depth=0, num_neur = 128, dropout=False):
        super(FullyConnectedRegularized, self).__init__()

        # self.l2_reg = l2_reg
        self.num_in_var = num_in_var

        self.layer_list = []
        fcIn = nn.Linear(self.num_in_var, num_neur)
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
        assert (x.shape[1] == self.num_in_var), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_in_var, x.shape[1])
        output = self.Layers(x)
        return output


class CNN(nn.Module):
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


        if dropout:
            self.cnn_layer_list.append(nn.Dropout(0.2))
        self.cnn_layer_list.append(nn.Conv1d(in_channels=1, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
        in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
        self.cnn_layer_list.append(nn.ReLU())
        self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
        self.cnn_layer_list.append(nn.MaxPool1d(kernel_size = max_pool_kernel_size, stride=max_pool_stride))
        in_lenght = np.floor(((in_lenght-(max_pool_kernel_size - 1) -1) / max_pool_stride)+1)
        if dropout:
            self.cnn_layer_list.append(nn.Dropout(0.5))
        for depth in range(num_depth):
            self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
            in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
            self.cnn_layer_list.append(nn.ReLU())
            self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
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

        self.CNN_Layers = nn.Sequential(*self.cnn_layer_list)
        self.FC_Layers = nn.Sequential(*self.fc_layer_list)

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_in_var + 1), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_in_var, x.shape[1])
        house_cons = x[:,1:].view(x.shape[0],-1,x.shape[1]-1)
        need = x[:,0].view(-1, x.shape[0]).t()

        cnn_output = self.CNN_Layers(house_cons)
        # print("shape of CNN output ", cnn_output.shape)
        cnn_output = cnn_output.view(x.shape[0], -1)
        # print("shape of CNN output ", cnn_output.shape)
        fc_input = torch.cat([need, cnn_output], dim = 1)
        # print("shape of CNN output ", fc_input.shape)
        fc_output = self.FC_Layers(fc_input)
        # print("shape of CNN output ", fc_output.shape)
        return fc_output

class CNN2(nn.Module):
    def __init__(self, num_in_var = 94, num_depth=0, num_neur = 128, dropout=False):
        super(CNN2, self).__init__()

        self.num_in_var = num_in_var
        # self.l2_reg = l2_reg
        in_lenght = num_in_var
        in_channel = 1
        min_in_fc = 4
        max_pool_stride = 2
        max_pool_kernel_size = 3
        cnn_stride = 1
        cnn_kernel_size = 3
        num_channel = 32
        num_out_var = 3*4
        self.cnn_layer_list = []
        self.fc_layer_list = []
        self.sigmoid = nn.Sigmoid()

        # self.bn1 = nn.BatchNorm2d(32)

        self.cnn_layer_list.append(nn.Conv1d(in_channels=1, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
        in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
        self.cnn_layer_list.append(nn.ReLU())
        self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))

        self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
        in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
        self.cnn_layer_list.append(nn.ReLU())
        self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))

        self.cnn_layer_list.append(nn.MaxPool1d(kernel_size = max_pool_kernel_size, stride=max_pool_stride))
        in_lenght = np.floor(((in_lenght-(max_pool_kernel_size - 1) -1) / max_pool_stride)+1)

        for depth in range(num_depth):

            self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
            in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
            self.cnn_layer_list.append(nn.ReLU())
            self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))

            self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
            in_lenght = np.floor(((in_lenght-(cnn_kernel_size - 1)-1) / cnn_stride)+1)
            self.cnn_layer_list.append(nn.ReLU())
            self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))


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

        self.CNN_Layers = nn.Sequential(*self.cnn_layer_list)
        self.FC_Layers = nn.Sequential(*self.fc_layer_list)

    def forward(self, x):
        #Forward pass
        assert (x.shape[1] == self.num_in_var + 1), "Wrong number of parameters\nnumber of parameters: {}\nsize of input: {}".format(
            self.num_in_var, x.shape[1])
        house_cons = x[:,1:].view(x.shape[0],-1,x.shape[1]-1)
        need = x[:,0].view(-1, x.shape[0]).t()

        cnn_output = self.CNN_Layers(house_cons)
        print("shape of CNN output ", cnn_output.shape)
        cnn_output = cnn_output.view(x.shape[0], -1)
        print("shape of CNN output ", cnn_output.shape)
        fc_input = torch.cat([need, cnn_output], dim = 1)
        print("shape of CNN output ", fc_input.shape)
        fc_output = self.FC_Layers(fc_input)
        print("shape of FC output ", fc_output.shape)
        fc_output = fc_output.view(x.shape[0], 3, 4)
        print("shape of FC output ", fc_output.shape)
        
        return self.sigmoid(fc_output)



# class CNNSimple(nn.Module):
#     def __init__(self, num_in_var = 94, num_out_var = 94, num_depth=0, num_neur = 128, dropout=False):
#         super(CNN, self).__init__()

#         self.num_in_var = num_in_var
#         # self.l2_reg = l2_reg
#         in_lenght = num_in_var
#         in_channel = 1
#         min_in_fc = 4
#         max_pool_stride = 2
#         max_pool_kernel_size = 3
#         cnn_stride = 1
#         cnn_kernel_size = 3
#         num_channel = 64
#         self.cnn_layer_list = []
#         self.fc_layer_list = []

#         # self.bn1 = nn.BatchNorm2d(32)


#         self.cnn_layer_list.append(nn.Conv1d(in_channels=1, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
#         self.cnn_layer_list.append(nn.ReLU())
#         self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))
#         for depth in range(num_depth):
#             self.cnn_layer_list.append(nn.Conv1d(in_channels=num_channel, out_channels=num_channel, kernel_size=cnn_kernel_size, stride=cnn_stride))
#             self.cnn_layer_list.append(nn.ReLU())
#             self.cnn_layer_list.append(nn.BatchNorm1d(num_channel))

#         self.CNN_Layers = nn.Sequential(*self.cnn_layer_list)

#     def forward(self, x):
#         # return fc_output