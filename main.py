import pandas as pd
import createdata
import numpy as np
import bp
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = 'D:\homework\picture\picall'
    createdata.createcsv(path)
    f = open('labels.txt', 'r')
    dataset = f.readlines()
    f.close()
    for i in range(len(dataset)):
        dataset[i] = dataset[i].rstrip().split(',')
        for j in range(5):
            dataset[i][j] = float(dataset[i][j])
    train_dataset = dataset[:210]
    test_dataset = dataset[210:]
    x_train = []
    y_train = []
    for data in train_dataset:
        x_train.append(data[:-1])
        if data[5][:1] == 'g':
            y_train.append([1, 0, 0])
        if data[5][:1] == 'n':
            y_train.append([0, 1, 0])
        if data[5][:1] == 'r':
            y_train.append([0, 0, 1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = []
    y_test = []
    for data in train_dataset:
        x_test.append(data[:-1])
        if data[5][:1] == 'g':
            y_test.append([1, 0, 0])
        if data[5][:1] == 'n':
            y_test.append([0, 1, 0])
        if data[5][:1] == 'r':
            y_test.append([0, 0, 1])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    net = bp.NNetWork([5, 19, 3])
    net.train(x_train, y_train)
    acc = bp.ComputeAccuracy(net, x_test, y_test)
    print('Accuracy is:', acc)
    plt.show()
