import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import time
import sys
import csv


import generateur_csv as g_csv
import log_writer as lw
import Dataset as ds
import Network as nw

# set to true to one once, then back to false unless you want to change something in your training data.
CREATE_CSV = True
PATH_DATA = "./../DATA/"
# CSV_NAME = "input.csv"
LOG_DIR = "./../log/"
MODEL_DIR = "model/"
BEST_MODELE = "best_model.pt"
# MODEL_PATH = LOG_DIR + FC1 + BEST_MODELE

# MODELE_LOG_FILE = LOG_DIR + "modele.log"
# MODELE_TIME = f"model-{int(time.time())}"
METRICS = "metrics/"
TENSORBOARD = "tensorboard/"
DIEZ = "##########"


#TODO: Add Scheduler
#TODO: Normalize input

# class Normalize(object):

#     def __call__(self, sample):
#         image = sample['X_image']

#         # landmarks = landmarks.transpose((2, 0, 1))
#         return {'X_image': (image/255) -0.5,
#                 'Y': sample['Y']}

# class CrossEntropyOneHot(object):


class ModelCheckpoint:
    """
    Model check class
    """

    def __init__(self, filepath, model):
        """Constructor

        Arguments:
            filepath {string} -- file path of the best model and where to save it
            model {pytorch model} -- model
        """
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print(DIEZ + " Saving a better model to {} ".format(self.filepath)+DIEZ)
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss
            return True
        return False


def progress(loss, acc, description="Training"):
    print(description +
          '   : Loss : {:2.4f}, Acc : {:2.4f}\r'.format(loss, acc))
    sys.stdout.flush()


def main():

    #Parsing the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1,
                        help="number of epoch (default: 1)")
    parser.add_argument("--batch", type=int, default=100,
                        help="number of batch (default: 100)")
    parser.add_argument("--valpct", type=float, default=0.2,
                        help="proportion of test data (default: 0.2)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="number of thread used (default: 1)")
    parser.add_argument("--create_csv", default=False, action='store_true',
                        help="create or not csv file (default: False)")
    parser.add_argument("--log", default=False, action='store_true',
                        help="Write log or not (default: False)")
    parser.add_argument("--l2_reg", type=float, default=0.001,
                        help="L2 regularisation (default: 0.001)")
    parser.add_argument("--dropout", default=False, action='store_true',
                        help="Activate or not dropout")

    parser.add_argument("--num_var", type=int, default=5,
                        help="Number of variables (default: 5)")
    parser.add_argument("--num_const", type=int, default=8,
                        help="number of constrains (default: 8)")
    parser.add_argument("--num_prob", type=int, default=10,
                        help="number of problems to generate (default: 10)")
    parser.add_argument("--loss", type = str, default="MSE",
                        help="Use of custom loss (default: MSE)")
    parser.add_argument("--num_deep_layer", type=int, default=1,
                        help="Number of deep layer used (default: 1)")
    parser.add_argument("--num_neur", type=int, default=100,
                        help="Number neurons in each layer (default: 100)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="constraint penalty (default: 0.0)")

    args = parser.parse_args()

    #Recreate a new dataset csv if necessary

    file_path = PATH_DATA + \
        "inputV{}C{}P{}".format(
            args.num_var, args.num_const, args.num_prob) + ".csv"
    if args.create_csv:
        g_csv.generate_csv(file_path, args.num_var,
                           args.num_const, args.num_prob)

    valid_ratio = args.valpct  # Going to use 80%/20% split for train/valid

    #Transformation of the dataset
    data_transforms = transforms.Compose([
        ds.ToTensor()  # transform to pytorch tensor
    ])

    #create a dataset object to facilitate streaming of data
    full_dataset = ds.LP_data(
        csv_file_name=file_path, transform=data_transforms)

    #number of elements taken for training and testing
    nb_train = int((1.0 - valid_ratio) * len(full_dataset))
    # nb_test = int(valid_ratio * len(full_dataset))
    nb_test = len(full_dataset) - nb_train

    #Print info
    print("Size of full data set: ", len(full_dataset))
    print("Size of training data: ", nb_train)
    print("Size of testing data: ", nb_test)

    #Splitting of data
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(
        full_dataset, [nb_train, nb_test])

    #Create data loader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch,
                              shuffle=True,
                              num_workers=args.num_threads)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch,
                             shuffle=True,
                             num_workers=args.num_threads)

    # for (inputs, targets) in train_loader:
    #     print("input:\n",inputs)
    #     print("target:\n", targets)

    #compute number of input parameters
    num_param = args.num_var + args.num_const + (args.num_var*args.num_const)

    #print info
    print("number of variables: ", args.num_var)
    print("number of const: ", args.num_const)
    print("Number of parameters: ", num_param)

    if args.num_deep_layer < 0:
        assert(False), "Not number of correct deep layers: {}".format(
            args.num_deep_layer)
    else:
        print("Model with {} layers".format(args.num_deep_layer))
        model = nw.FullyConnectedRegularized(
            num_param=num_param, num_var=args.num_var, num_depth=args.num_deep_layer,num_neur= args.num_neur, dropout=args.dropout)

    #print model info
    print("Network architechture:\n", model)
    # for name, param in model.named_parameters():
    #     print("name of parameters ",name)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    #Define loss
    # f_loss = torch.nn.CrossEntropyLoss()
    if args.loss == "MSE":
        print("MSE loss used with alpha: {}".format(args.alpha))
        loss_name = "MSELoss/"
        # f_loss = nn.MSELoss()
        f_loss = nw.CustomMSELoss(alpha = args.alpha, num_const=args.num_const)
    elif args.loss == "PCL":
        print("PCL used with alpha: {}".format(args.alpha))
        loss_name = "PCL/"
        f_loss = nw.PureCostLoss(alpha=args.alpha, num_const=args.num_const)
    elif args.loss == "GCL":
        print("GCL used with alpha: {}".format(args.alpha))
        loss_name = "GCL/"
        f_loss = nw.GuidedCostLoss(alpha=args.alpha, num_const=args.num_const)


    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2_reg)

    #Make run directory
    run_name = "runV{}C{}D{}L{}-".format(
        args.num_var, args.num_const, args.num_deep_layer, args.loss)
    run_dir_path, num_run = lw.generate_unique_dir(LOG_DIR, run_name)

    #setup model checkpoint
    path_model_check_point = run_dir_path + MODEL_DIR
    if not os.path.exists(path_model_check_point):
        os.mkdir(path_model_check_point)
    model_checkpoint = ModelCheckpoint(
        path_model_check_point + BEST_MODELE, model)

    #setup logging
    if args.log:
        print("Writing log")
        #generate unique folder for new run
        tensorboard_writer = SummaryWriter(
            log_dir=run_dir_path, filename_suffix=".log")

        #write short description of the run
        run_desc = "Epoch{}V{}Cst{}Loss{}Dlayer{}Alpha{}".format(
            args.epoch, args.num_var, args.num_const, args.loss, args.num_deep_layer, args.alpha)
        log_file_path = LOG_DIR + "Run{}".format(num_run) + run_desc + ".log"

        lw.summary_writer(run_dir_path, model, optimizer,
                          tensorboard_writer, num_run)

    last_update = 0
    start_time = time.time()
    with tqdm(total=args.epoch) as pbar:
        for t in range(args.epoch):
            pbar.update(1)
            pbar.set_description("Epoch Number {}".format(t))
            # print("\n\n",DIEZ + "Epoch Number: {}".format(t) + DIEZ)

            #train
            train_loss, train_acc, train_cost, train_penalty = nw.train(
                model, train_loader, f_loss, optimizer, device)

            progress(train_loss, train_acc, description="Trainning")
            time.sleep(0.5)
            # print(args.custom_loss)

            #test
            val_loss, val_acc, val_cost, val_penalty, _ = nw.test(
                model, test_loader, f_loss, device)

            progress(val_loss, val_acc, description="Validation")
            # print("\n\n","Validation : Loss : {:.4f}, Acc : {:.4f}".format(
            #     val_loss, val_acc))

            #check if model is best and save it if it is
            if model_checkpoint.update(val_loss):
                last_update = 0
            else:
                last_update += 1
                print("Last update: ", last_update)

            if args.log:

                #Write tensorboard and log
                tensorboard_writer.add_scalars(loss_name, {'train_loss': train_loss,
                                                           'val_loss': val_loss
                                                           }, t)

                tensorboard_writer.add_scalars('PureCost/', {
                    'train_cost': train_cost,
                    'val_cost': val_cost
                }, t)
                tensorboard_writer.add_scalars('Penalty/', {
                    'train_penalty': train_penalty,
                    'val_penalty': val_penalty
                }, t)

                lw.write_log(log_file_path, val_acc,
                             val_loss, train_acc, train_loss)
    total_run_time = time.time() - start_time
    print("--- %s seconds ---" % (total_run_time))

    model.load_state_dict(torch.load(path_model_check_point + BEST_MODELE))
    print(DIEZ+" Final Test "+DIEZ)

    test_loss, test_acc, test_cost, test_penalty, example_text = nw.test(
        model, test_loader, f_loss, device, final_test=True, num_const= args.num_const)
    print("Test       : Loss : {:.4f}, Acc : {:.4f}".format(
        test_loss, test_acc))
    print("Test       : Cost : {:.4f}, Pen : {:.4f}".format(
        test_cost, test_penalty))

    if args.log:
        lw.end_summary_witer(run_dir_path, total_run_time, test_loss,
                             test_acc, test_cost, test_penalty, tensorboard_writer, num_run)
        lw.write_examples(run_dir_path, example_text,
                          tensorboard_writer, num_run)
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
