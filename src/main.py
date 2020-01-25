import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
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
CSV_NAME = "input.csv"
LOG_DIR = "./../log/"
FC1 = "fc1/"
BEST_MODELE = "best_model.pt"
MODEL_PATH = LOG_DIR + FC1 + BEST_MODELE

# MODELE_LOG_FILE = LOG_DIR + "modele.log"
# MODELE_TIME = f"model-{int(time.time())}"
METRICS = "metrics/"
TENSORBOARD = "tensorboard/"
DIEZ = "##########"
# tensorboard_writer   = SummaryWriter(log_dir = LOG_DIR+TENSORBOARD)





# class Normalize(object):

#     def __call__(self, sample):
#         image = sample['X_image']

#         # landmarks = landmarks.transpose((2, 0, 1))
#         return {'X_image': (image/255) -0.5,
#                 'Y': sample['Y']}

# class CrossEntropyOneHot(object):


#     def __call__(self, sample):
#         _, labels = sample['Y'].max(dim=0)
#         # landmarks = landmarks.transpose((2, 0, 1))
#         return {'X_image': sample['X_image'],
#                 'Y': labels}





class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model to ", self.filepath)
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss


def progress(loss, acc, description="Training"):
    print(description +
          '   : Loss : {:2.4f}, Acc : {:2.4f}\r'.format(loss, acc))
    sys.stdout.flush()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1,
                        help="number of epoch (default: 1)")
    parser.add_argument("--batch", type=int, default=100,
                        help="number of batch (default: 100)")
    parser.add_argument("--valpct", type=float, default=0.2,
                        help="proportion of test data (default: 0.2)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="number of thread used (default: 1)")
    parser.add_argument("--create_csv", type=bool, default=False,
                        help="create or not csv file (default: False)")
    parser.add_argument("--l2_reg", type=int, default=0.001,
                        help="L2 regularisation (default: 0.001)")

    parser.add_argument("--num_var", type=int, default=5,
                        help="Number of variables (default: 5)")
    parser.add_argument("--num_const", type=int, default=4,
                        help="number of constrains (default: 4)")
    parser.add_argument("--num_prob", type=int, default=10,
                        help="number of problems to generate (default: 10)")

    args = parser.parse_args()

    if args.create_csv:
        g_csv.generate_csv(PATH_DATA + CSV_NAME, args.num_var,
                           args.num_const, args.num_prob)
        #TODO

    valid_ratio = args.valpct  # Going to use 80%/20% split for train/valid

    data_transforms = transforms.Compose([
        ds.ToTensor()
    ])

    #TODO
    full_dataset = ds.LP_data(
        csv_file_name=PATH_DATA + CSV_NAME, transform=data_transforms)

    nb_train = int((1.0 - valid_ratio) * len(full_dataset))
    # nb_test = int(valid_ratio * len(full_dataset))
    nb_test = len(full_dataset) - nb_train
    print("Size of full data set: ", len(full_dataset))
    print("Size of training data: ", nb_train)
    print("Size of testing data: ", nb_test)
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(
        full_dataset, [nb_train, nb_test])

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

    #TODO params
    num_param = args.num_var + args.num_const + (args.num_var*args.num_const)
    print("Number of parameters: ", num_param)
    model = nw.FullyConnectedRegularized(
        l2_reg=args.l2_reg, num_param=num_param, num_var=args.num_var)
    print("Network architechture:\n", model)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    # f_loss = torch.nn.CrossEntropyLoss() #TODO
    f_loss = nn.MSELoss()
    f_loss_custom = nw.CustomLoss(args.num_const)
    optimizer = torch.optim.Adam(model.parameters())

    top_logdir = LOG_DIR + FC1
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)
    model_checkpoint = ModelCheckpoint(top_logdir + BEST_MODELE, model)

    log_file_path = lw.generate_unique_logpath(LOG_DIR, "Linear")
    with tqdm(total=args.epoch) as pbar:
        for t in range(args.epoch):
            pbar.update(1)
            pbar.set_description("Epoch Number{}".format(t))
            # print("\n\n",DIEZ + "Epoch Number: {}".format(t) + DIEZ)
            train_loss, train_acc = nw.train(
                model, train_loader, f_loss, optimizer, device)

            progress(train_loss, train_acc, description="Trainning")
            time.sleep(0.5)

            val_loss, val_acc = nw.test(model, test_loader, f_loss, device)

            progress(val_loss, val_acc, description="Validation")
            # print("\n\n","Validation : Loss : {:.4f}, Acc : {:.4f}".format(
            #     val_loss, val_acc))

            model_checkpoint.update(val_loss)

            lw.write_log(log_file_path, val_acc,
                         val_loss, train_acc, train_loss)

            # tensorboard_writer.add_scalar(METRICS + 'train_loss', train_loss, t)
            # tensorboard_writer.add_scalar(METRICS + 'train_acc',  train_acc, t)
            # tensorboard_writer.add_scalar(METRICS + 'val_loss', val_loss, t)
            # tensorboard_writer.add_scalar(METRICS + 'val_acc',  val_acc, t)

    model.load_state_dict(torch.load(MODEL_PATH))
    print(DIEZ+" Final Test "+DIEZ)
    test_loss, test_acc = nw.test(
        model, test_loader, f_loss, device, final_test=True)
    print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(
        test_loss, test_acc))

    # lw.create_acc_loss_graph(log_file_path)


if __name__ == "__main__":
    main()