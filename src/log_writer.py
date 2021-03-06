import time
import matplotlib.pyplot as plt
from matplotlib import style
import os
import sys
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import cv2


style.use("ggplot")
# MAX_TIME = 1000000


class LogManager:
    """
    Log manager class
    """
    def __init__(self, logdir, raw_run_name):
        self.logdir = logdir
        self.raw_run_name = raw_run_name
        self.run_num = 0
        self.example_text = ""

    def set_tensorboard_writer(self, tensorboard_writer):
        self.tensorboard_writer = tensorboard_writer

    def generate_unique_dir(self):
        """Generate unique dir path
        
        Returns:
            [str] -- [new path to a directory]
        """
        i = 0
        while(True):
            # i = int(time.time() % MAX_TIME)
            i = int(time.time())
            run_name = self.raw_run_name + str(i)
            run_folder = os.path.join(self.logdir, run_name)
            if not os.path.isdir(run_folder):
                print("New run folder: {}".format(run_folder))
                os.mkdir(run_folder)
                self.run_num = i
                self.run_dir_path = run_folder
                return run_folder + "/", i
            # i = i + 1
            time.sleep(1)

    def generate_unique_logpath(self):
        """generate new logpath
        
        Returns:
            str -- new log path
        """
        i = 0
        while(True):
            # i = int(time.time() % MAX_TIME)
            i = int(time.time())
            run_name = self.raw_run_name + str(i)
            log_path = os.path.join(self.logdir, run_name + ".log")
            if not os.path.isfile(log_path):
                print("New log file: {}".format(log_path))
                return log_path
            time.sleep(1)

    def write_log(self, log_file_path, val_acc, val_loss, train_acc, train_loss):
        """ 
        Write log
        """
        with open(log_file_path, "a") as f:
            print("Logging to {}".format(log_file_path))
            f.write(f"{round(time.time(),3)},{round(float(val_acc),2)},{round(float(val_loss),4)},{round(float(train_acc),2)},{round(float(train_loss),4)}\n")

    def create_acc_loss_graph(self, log_file_path):
        contents = open(log_file_path, "r").read().split("\n")

        list_time = []
        list_train_acc = []
        list_train_loss = []

        list_val_acc = []
        list_val_loss = []

        for c in contents:
            if "," in c:
                timestamp, val_acc, val_loss, train_acc, train_loss = c.split(
                    ",")

                list_time.append(float(timestamp))

                list_val_acc.append(float(val_acc))
                list_val_loss.append(float(val_loss))

                list_train_acc.append(float(train_acc))
                list_train_loss.append(float(train_loss))

        fig = plt.figure()

        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

        ax1.plot(list_time, list_train_acc, label="train_acc")
        ax1.plot(list_time, list_val_acc, label="val_acc")
        ax1.legend(loc=2)
        ax2.plot(list_time, list_train_loss, label="train_loss")
        ax2.plot(list_time, list_val_loss, label="val_loss")
        ax2.legend(loc=2)
        plt.show()

    def write_examples(self):
        """
        Write example text in log and in tensor board
        """
        example_file = open(self.run_dir_path + "/examples.txt", 'w')
        example_file.write(self.example_text)
        example_file.close()
        self.tensorboard_writer.add_text(
            "Run {} Example".format(self.run_num), self.example_text)

    def summary_writer(self, model, optimizer):
        """
        Write summary of the run
        """

        summary_file = open(self.run_dir_path + "/summary.txt", 'w')

        summary_text = """
    RUN Number: {}
    ===============

    Executed command
    ================
    {}

    Model summary
    =============
    {}


    {} trainable parameters

    Optimizer
    ========
    {}
    """.format(self.run_num, " ".join(sys.argv), model, sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer)
        summary_file.write(summary_text)
        summary_file.close()

        self.tensorboard_writer.add_text(
            "Run {} Summary".format(self.run_num), summary_text)

    def end_summary_witer(self, total_train_time, test_loss, test_acc, test_cost, test_pen):
        """
        Write end summary
        """

        summary_file = open(self.run_dir_path + "/summary.txt", 'a')
        summary_text = """

    Total training time
    ================
    {}

    Best Model validation
    =============
    Loss:    {:.4f}, Acc:    {:.4f}
    Cost:    {:.4f}, Pen:    {:.4f}

    """.format(total_train_time, test_loss, test_acc, test_cost, test_pen)
        summary_file.write(summary_text)
        summary_file.close()

        self.tensorboard_writer.add_text(
            "Run {} Summary".format(self.run_num), summary_text)



class LP_Log(LogManager):
    """
    Log for LP problem
    """
    def __init__(self, logdir, raw_run_name, num_const=0):
        super().__init__(logdir, raw_run_name)
        self.num_const = num_const

    def write_example(self, num, outputs, targets, inputs):
        """
        Write result of test
        """

        num_batch = outputs.shape[0]
        num_var = outputs.shape[1]
        output_cost = torch.bmm(outputs.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))
        target_cost = torch.bmm(targets.view(
            num_batch, 1, num_var), inputs[:, -num_var:].view(num_batch, num_var, 1))

        a_const = inputs[:, :-self.num_const -
                         num_var].view(num_batch, self.num_const, num_var).transpose(2, 1)
        b_const = inputs[:, -self.num_const -
                         num_var:-num_var].view(num_batch, 1, -1)

        output_penalty = (torch.clamp((torch.bmm(outputs.view(
            num_batch, 1, num_var), a_const) - b_const), min=0)).sum(dim=2)

        negative_penalty = nn.ReLU()(-outputs).sum(dim=1).view(num_batch, 1)

        output_penalty += negative_penalty

        A = (a_const.transpose(2, 1)[0]).tolist()
        B = (b_const[0]).tolist()
        C = (inputs[:, -num_var:][0]).tolist()

        txt = """Example {}
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
        self.example_text += txt
        print("example_text: ", txt)

        # return example_text


class EDF_Log(LogManager):
    def __init__(self, logdir, raw_run_name):
        super().__init__(logdir, raw_run_name)

    def plot_writer(self, outputs, targets, inputs, num):
        image_folder = self.run_dir_path + "/images/"
        if not os.path.isdir(image_folder):
            os.mkdir(image_folder)
        n = 1
        output_pve = (outputs[:n, :].tolist()[0])
        target_pve = (targets[:n, :].tolist()[0])
        house_cons = (inputs[:n, 1:].tolist()[0])
        fig = plt.figure()
        plt.plot(output_pve[:], label="output_pve")
        plt.plot(target_pve[:], label="target_pve")
        plt.plot(house_cons[:], label="house_cons")
        plt.title('Courbe{}'.format(num))
        plt.legend()
        plt.savefig(image_folder + 'Courbe{}.png'.format(num),
                    bbox_inches='tight')
        im = cv2.imread(image_folder + 'Courbe{}.png'.format(num))
        im = im.transpose((2, 0, 1))
        self.tensorboard_writer.add_image(
            "Run_{}_Results/Courbe{}".format(self.run_num, num), im)
        
    def write_example(self, num, outputs, targets, inputs):
        num_batch = outputs.shape[0]
        house_cons = inputs[:, 1:]
        self.plot_writer(outputs, targets, inputs, num)
