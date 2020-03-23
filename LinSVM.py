import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class LinearSVM(object):
    def __init__(self, epochs):
        self.x = pd.DataFrame()
        self.y = np.array([])
        self.weights = np.array([])
        self.epochs = epochs
        self.reg_strength = 10
        self.learning_rate = 0.001
        self.lamda = 1

    def hingeloss(self, d, n):
        return self.reg_strength*(np.sum(d)/n)

    def get_cost(self, w, x, y):
        n = x.shape[0]
        dist = 1 - y*(np.dot(x, w))
        dist[dist < 0] = 0
        cost = (self.lamda/2)*(np.dot(w, w)) + self.hingeloss(dist, n)
        return cost

    def get_cost_gradient(self, W, X_batch, Y_batch):
        dist = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))
        if max(0, dist) == 0:
            di = W
        else:
            di = W - (self.reg_strength * Y_batch * X_batch)
            dw += di
        return dw

    def initialize(self, x, y):
        self.x = x
        self.y = y
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
        prev_cost = float("inf")
        cost_threshold = 0.001

        for epoch in range(1, max_epochs):
            X, Y = shuffle(features, outputs)
            for index, x in enumerate(X):
                gradient = self.get_cost_gradient(self.weights, x, Y[index])
                self.weights = self.weights - (self.learning_rate * gradient)
           
            if(epoch%3 == 0):           
                cost = self.get_cost(self.weights, features, outputs)
                print("Iteration: " + str(epoch) + "\tcost: " + str(cost))
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    break
                prev_cost = cost

    def train(self, x, y):
        x, y = self.initialize(x, y)
        print("training started...")
        self.sgd(x, y)
        print("training finished\n")
        y_pred = np.sign(np.dot(x, self.weights))
        self.PerformanceMatrix(x, y, y_pred)
        return y_pred

    def test(self, x, y):
        x = self.featureSelect(x)
        y = np.array(y)
        x = np.array(x)
        t = np.ones((np.size(x, 0), 1))
        x = np.append(x, t, axis=1)
        y_pred = np.sign(np.dot(x, self.weights))
        self.PerformanceMatrix(x, y, y_pred)
        return y_pred

    def featureSelect(self, x):
        mask = list(self.x.columns.values)
        x = x.loc[:, mask]
        x = x.to_numpy()
        return x

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


def readfile(filename):
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

def calculate_accuracy(y_test, y_pred):
    mis = 0
    for Y, Y_pred in zip(y_test, y_pred):
        if(Y != Y_pred):
            mis += 1
    accuracy = (float(len(y_test)) - mis)/float(len(y_test))
    print("\nFinal accuracy: " + str(accuracy))
    print("Total misclassifications: " + str(mis) +
          " (Out of " + str(len(y_test)) + ")")

# Binary Classification

# x, y = readfile('Dataset/a4a.txt')
# inference = LinearSVM(epochs=128)
# y_pred_train = inference.train(x, y)
# print("\nTesting...")
# x_test, y_test = readfile('Dataset/a4a_t.txt')
# t_pred_test = inference.test(x_test, y_test)

# Multiclass Classification

x, y = readfile('Dataset/iris.txt')
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.75, random_state=42
)

y_train_1 = np.array(y_train)
y_train_2 = np.array(y_train)
y_test_1 = np.array(y_test)
y_test_2 = np.array(y_test)

y_train_1[y_train_1 >= 2.0] = -1.0
y_train_2[y_train_2 != 3.0] = -1.0
y_train_2[y_train_2 == 3.0] = 1.0

y_test_1[y_test_1 >= 2.0] = -1.0
y_test_2[y_test_2 != 3.0] = -1.0
y_test_2[y_test_2 == 3.0] = 1.0

classifier1 = LinearSVM(epochs=128)
classifier2 = LinearSVM(epochs=128)
print("\nTraining classifier 1")
y_pred_train_1 = classifier1.train(x_train, y_train_1)
print("\nTraining classifier 2")
y_pred_train_2 = classifier2.train(x_train, y_train_2)
print("\nTesting...")
print("\nTesting classifier 1")
y_pred_test_1 = classifier1.test(x_test, y_test_1)
print("\nTesting classifier 2")
y_pred_test_2 = classifier2.test(x_test, y_test_2)

y_pred = []
for clf1, clf2 in zip(y_pred_test_1, y_pred_test_2):
    if (clf1 == 1):
        y_pred.append(1.0)
    elif (clf2 == 1):
        y_pred.append(3.0)
    else:
        y_pred.append(2.0)

calculate_accuracy(y_test, y_pred)
