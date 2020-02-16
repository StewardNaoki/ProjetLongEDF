import time
import matplotlib.pyplot as plt
from matplotlib import style
import os
import sys
import csv
import pandas as pd

style.use("ggplot")
# MAX_TIME = 1000000

class LogManager:
    def __init__(self, logdir, raw_run_name):
        self.logdir = logdir
        self.raw_run_name = raw_run_name
        self.run_num = 0
    
    def set_tensorboard_writer(self, tensorboard_writer):
        self.tensorboard_writer = tensorboard_writer

    def generate_unique_dir(self):
        i = 0
        while(True):
            # i = int(time.time() % MAX_TIME)
            i = int(time.time())
            run_name = self.raw_run_name  + str(i)
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


    def write_log(self,log_file_path, val_acc, val_loss, train_acc, train_loss):
        with open(log_file_path, "a") as f:
            print("Logging to {}".format(log_file_path))
            f.write(f"{round(time.time(),3)},{round(float(val_acc),2)},{round(float(val_loss),4)},{round(float(train_acc),2)},{round(float(train_loss),4)}\n")




    def create_acc_loss_graph(self,log_file_path):
        contents = open(log_file_path, "r").read().split("\n")

        list_time = []
        list_train_acc = []
        list_train_loss = []

        list_val_acc = []
        list_val_loss = []

        for c in contents:
            if "," in c:
                timestamp, val_acc, val_loss, train_acc, train_loss = c.split(",")

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

    def write_examples(self,example_text):
        example_file = open(self.run_dir_path + "/examples.txt", 'w')
        example_file.write(example_text)
        example_file.close()
        self.tensorboard_writer.add_text("Run {} Example".format(self.run_num), example_text)

    def summary_writer(self,model, optimizer):

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

        self.tensorboard_writer.add_text("Run {} Summary".format(self.run_num), summary_text)


    def end_summary_witer(self, total_train_time, test_loss, test_acc, test_cost, test_pen):

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

        self.tensorboard_writer.add_text("Run {} Summary".format(self.run_num), summary_text)


def plot_writer(outputs, targets, inputs):
    n = min(15, len(outputs))
    output_pve = outputs[:n, :].tolist()
    target_pve = targets[:n, :].tolist()
    house_cons = inputs[:n, 3:].tolist()
    # output_pve = outputs[:n, :]
    # target_pve = targets[:n, :]
    # house_cons = inputs[:n, 3:]
    # print(output_pve)
    # print(target_pve)
    # print(house_cons)
    dict_csv = {"output_pve": [], "target_pve": [], "house_cons": []}
    for k in range(n):
        dict_csv["output_pve"].append(output_pve[k][:])
        dict_csv["target_pve"].append(target_pve[k][:])
        dict_csv["house_cons"].append(house_cons[k][:])
    df_csv = pd.DataFrame(dict_csv)
    print(df_csv.head())
    df_csv.to_csv('test.csv', index=False)

def plot_reader():
    data_frame = pd.read_csv('test.csv')
    # print(data_frame.head())
    output_pve = data_frame["output_pve"].iloc[1]
    target_pve = data_frame["target_pve"].iloc[1]
    house_cons = data_frame["house_cons"].iloc[1]
    # print(type(output_pve))
    # print(target_pve)
    # print(house_cons)
    for k in range(data_frame.shape[0]):
        output_pve = eval(data_frame["output_pve"].iloc[k])
        target_pve = eval(data_frame["target_pve"].iloc[k])
        house_cons = eval(data_frame["house_cons"].iloc[k])
        for i in range(len(output_pve)):
            print(output_pve[i])
            print(target_pve[i])
            print(house_cons[i])
