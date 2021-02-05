import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    train_file = open("../checkpoint/BCNN_each_epoch_record_train.txt", 'r')
    test_file = open('../checkpoint/BCNN_each_epoch_record_val.txt', 'r')

    train_total = train_file.readlines()
    test_total = test_file.readlines()
    train_num = len(train_total)
    test_num = len(test_total)
    train_idx = np.arange(train_num)
    trian_acc_list = []
    trian_loss_list = []
    test_acc_list = []
    test_loss_list = []


    for idx in range(train_num):
        train_str = train_total[idx].split(',')
        test_str = test_total[idx].split(',')

        train_acc = float(train_str[1].split(':')[-1])
        test_acc = float(test_str[1].split(':')[-1])

        trian_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        train_loss = float(train_str[2].split(':')[-1])
        test_loss = float(test_str[2].split(':')[-1])
        trian_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_idx, trian_acc_list, label='train_acc')
    ax.plot(train_idx, test_acc_list, label='test_acc')
    ax.legend()
    plt.savefig('acc.png')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_idx, trian_loss_list, label='train_loss')
    ax.plot(train_idx, test_loss_list, label='test_loss')
    ax.legend()
    plt.savefig('loss.png')
    plt.show()

