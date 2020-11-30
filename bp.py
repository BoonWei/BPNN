import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NNetWork:
    def __init__(self, cell_num_list):
        """
        :param cell_num_list: a list,each element is cell number of every layer.
        For example,[4, 10, 3] means 3 layers,4 cells in input layer,10 cells in hidden layer and 3 cells in output layer
        """
        self.__layer_num = len(cell_num_list) - 1
        self.__weight_list = []
        self.__bias_list = []
        for i in range(self.__layer_num):
            weight = np.random.normal(size=(cell_num_list[i], cell_num_list[i + 1]))
            bias = np.zeros((1, cell_num_list[i + 1]))
            self.__weight_list.append(weight)
            self.__bias_list.append(bias)

    def predict(self, x):
        if len(x.shape) == 1:
            x.reshape(-1, len(x))
        a = x
        for i in range(self.__layer_num):
            z = np.dot(a, self.__weight_list[i]) + self.__bias_list[i]
            a = sigmoid(z)
        return a

    def __CostGradient(self, x, y):
        """
        Compute the cost and gradient using input and label.
        :param x: input
        :param y: label
        :return: cost and a list of gradient of cost function in each layer
        """
        a_list = []
        a = x
        for i in range(self.__layer_num):
            a_list.append(a)
            z = np.dot(a, self.__weight_list[i]) + self.__bias_list[i]
            a = sigmoid(z)
        cost = np.sum((y - a) ** 2) / 2 / len(a)
        delta_list = []
        delta = (a - y) * a * (1 - a)
        for i in range(self.__layer_num):
            delta_list.append(delta)
            delta = np.dot(delta, self.__weight_list[-1 - i].T) * a_list[-1 - i] * (1 - a_list[-1 - i])
        delta_list.reverse()
        bias_gradient_list = []
        for i in range(self.__layer_num):
            bias_gradient = np.sum(delta_list[i], axis=0, keepdims=True) / len(x)
            bias_gradient_list.append(bias_gradient)
        weight_gradient_list = []
        for i in range(self.__layer_num):
            weight_gradient = np.dot(a_list[i].T, delta_list[i]) / len(x)
            weight_gradient_list.append(weight_gradient)
        return cost, weight_gradient_list, bias_gradient_list

    def train(self, x, y, alpha=0.6, iteration_num=20000):
        """
        Train the neural network.
        :param x: inputs of train dataset
        :param y: labels of train dataset
        :param alpha: learning rate
        :param iteration_num: the maximum iteration number
        """
        cost_list = []
        for _ in range(iteration_num):
            cost, weight_gradient_list, bias_gradient_list = self.__CostGradient(x, y)
            cost_list.append(cost)
            for i in range(self.__layer_num):
                self.__weight_list[i] -= alpha * weight_gradient_list[i]
                self.__bias_list[i] -= alpha * bias_gradient_list[i]
        plt.plot(range(iteration_num), cost_list), plt.xlabel('iteration number'), plt.ylabel('cost')


def ComputeAccuracy(net, x_test, y_test):
    """
    Compute the accuracy of net
    :param net: the neural network
    :param x_test: inputs of test dataset
    :param y_test: labels of test dataset
    :return: the accuracy in test dataset
    """
    pred = net.predict(x_test)
    temp = 0
    row, column = pred.shape
    for i in range(row):
        for j in range(column):
            if pred[i, j] >= 0.5:
                pred[i, j] = 1
            else:
                pred[i, j] = 0
        if (pred[i] == y_test[i]).all():
            temp += 1
    acc = temp / row
    return acc

