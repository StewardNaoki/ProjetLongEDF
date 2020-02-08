import time
import matplotlib.pyplot as plt
from matplotlib import style
import os
import sys

style.use("ggplot")
# MAX_TIME = 1000000

def generate_unique_dir(logdir, raw_run_name):
    i = 0
    while(True):
        # i = int(time.time() % MAX_TIME)
        i = int(time.time())
        run_name = raw_run_name  + str(i)
        run_folder = os.path.join(logdir, run_name)
        if not os.path.isdir(run_folder):
            print("New run folder: {}".format(run_folder))
            os.mkdir(run_folder)
            return run_folder + "/", i
        # i = i + 1
        time.sleep(1)


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        # i = int(time.time() % MAX_TIME)
        i = int(time.time())
        run_name = raw_run_name + str(i)
        log_path = os.path.join(logdir, run_name + ".log")
        if not os.path.isfile(log_path):
            print("New log file: {}".format(log_path))
            return log_path
        time.sleep(1)


def write_log(log_file_path, val_acc, val_loss, train_acc, train_loss):
    with open(log_file_path, "a") as f:
        print("Logging to {}".format(log_file_path))
        f.write(f"{round(time.time(),3)},{round(float(val_acc),2)},{round(float(val_loss),4)},{round(float(train_acc),2)},{round(float(train_loss),4)}\n")




def create_acc_loss_graph(log_file_path):
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

def write_examples(logdir, example_text, tensorboard_writer, num_run):
    example_file = open(logdir + "/examples.txt", 'w')
    example_file.write(example_text)
    example_file.close()
    tensorboard_writer.add_text("Run {} Example".format(num_run), example_text)




def summary_writer(logdir, model, optimizer, tensorboard_writer, num_run):

    summary_file = open(logdir + "/summary.txt", 'w')

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
""".format(num_run, " ".join(sys.argv), model, sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer)
    summary_file.write(summary_text)
    summary_file.close()

    tensorboard_writer.add_text("Run {} Summary".format(num_run), summary_text)


def end_summary_witer(logdir, total_train_time, test_loss, test_acc, test_cost, test_pen, tensorboard_writer, num_run):

    summary_file = open(logdir + "/summary.txt", 'a')
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

    tensorboard_writer.add_text("Run {} Summary".format(num_run), summary_text)
