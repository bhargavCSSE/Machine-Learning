import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


class LinearSVM(object):
    def __init__(self, epochs):
        self.x = pd.DataFrame()
        self.y = np.array([])
        self.weights = np.array([])
        self.epochs = epochs
        self.reg_strength = 10000
        self.learning_rate = 0.000001
        self.lamda = 1

    def hingeloss(self, d, n):
        return self.reg_strength*(np.sum(d)/n)

    def get_cost(self, w, x, y):
        n = x.shape[0]
        dist = 1 - y*(np.dot(x, w))
        dist[dist < 0] = 0
        loss = self.hingeloss(dist, n)
        cost = (self.lamda/2)*(np.dot(w, w)) + loss
        return cost

    def get_cost_gradient(self, W, X_batch, Y_batch):
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])
        dist = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))
        for index, d in enumerate(dist):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.reg_strength * Y_batch[index] * X_batch[index])
            dw += di
        dw = dw/len(Y_batch)
        return dw

    def initialize(self, filename):
        x, y = self.readfile(filename)
        y = np.array(y)
        x = np.array(x)
        t = np.ones((np.size(x, 0), 1))
        x = np.append(x, t, axis=1)
        print(x.shape)
        self.weights = np.zeros(np.size(x, 1))
        print(self.weights.shape)
        return x, y

    def sgd(self, features, outputs):
        max_epochs = self.epochs
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01
        # stochastic gradient descent
        for epoch in range(1, max_epochs):
            # shuffle to prevent repeating update cycles
            X, Y = shuffle(features, outputs)
            for index, x in enumerate(X):
                ascent = self.get_cost_gradient(self.weights, x, Y[index])
                self.weights = self.weights - (self.learning_rate * ascent)
            # convergence check on 2^nth epoch
            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.get_cost(self.weights, features, outputs)
                print("Epoch is:{} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return self.weights
                prev_cost = cost
                nth += 1
        return self.weights

    def train(self, filename):
        x, y = self.initialize(filename)
        print("training started...")
        W = self.sgd(x, y)
        print("training finished.")
        print("weights are: {}".format(W))
        prediction = []
        for i in range(len(y)):
            y_pred = np.sign(np.dot(x, self.weights))
            prediction.append(y_pred)
        print("Training accuracy: " + str(accuracy_score(y, prediction)))
        # self.PerformanceMatrix(x, y, prediction)

    def readfile(self, filename):
        x = []
        y = []
        data = open(filename)
        for index, line in enumerate(data):
            line = line.split(None, 1)
            if len(line) == 1:
                line += ['']
            label, features = line
            y.append(float(label))
            temp_x = {}
            for elem in features.split(None):
                name, value = elem.split(':')
                temp_x[int(name)] = (float(value))
            x = x + [temp_x]
        x = pd.DataFrame(x).fillna(-1)
        return x, y

    def PerformanceMatrix(self, X, y_actual, y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for test_instance_result, label in zip(y_pred, y_actual):
            if ((test_instance_result > 0.0) and (label > 0.0)):
                tp += 1
            if ((test_instance_result <= 0.0) and (label <= 0.0)):
                tn += 1
            if ((test_instance_result > 0.0) and (label <= 0.0)):
                fp += 1
            if ((test_instance_result <= 0.0) and (label > 0.0)):
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn + 0.00001)
        precision = tp / (tp + fp + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        print("Accuracy:     ", accuracy)
        print("Recall:       ", recall)
        print("Precision:    ", precision)
        print("F1:           ", f1)


inference = LinearSVM(5000)
inference.train('Dataset/a4a.txt')
