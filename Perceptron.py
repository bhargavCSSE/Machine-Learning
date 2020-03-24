import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
class Perceptron(object):
    def __init__(self, epochs):
        self.epochs = epochs
        self.w = np.array([])
        self.iter = 0
        self.mis = 0
        self.x = pd.DataFrame()
        self.y = np.array([])

    def Initialize(self, x, y):
        self.x = x
        self.y = y
        x = np.array(x)
        y = np.array(y)
        self.w = np.zeros(np.size(x, 1)+1)
        t = np.ones((np.size(x, 0), 1))
        x = np.append(x, t, axis=1)
        return x, y

    def predict(self, activation):
        if activation >= 0.0:
            return 1.0
        else:
            return -1.0

    def featureSelect(self, x):
        mask = list(self.x.columns.values)
        x = x.loc[:, mask]
        x = x.to_numpy()
        return x

    def testingMetrics(self, x, printTest=False):
        if printTest is True:
            for i in range(0, np.size(x, 0)):
                y_pred = self.predict(np.dot(x[i, :], np.transpose(self.w)))
                print(str(x[i]) + " " + str(y_pred))
        total = np.size(x, 0)
        accuracy = (total-self.mis)/total
        print("Training accuracy: " + str(accuracy))

    def PerformanceMatrix(self, y_actual, y_pred):
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

    def train(self, x, y):
        x, y = self.Initialize(x, y)
        while(True):
            x, y = shuffle(x, y)
            self.mis = 0
            predictions = []
            for i in range(0, len(y)):
                y_pred = self.predict(np.dot(x[i, :], np.transpose(self.w)))
                if (y[i]*y_pred) <= 0:
                    self.w = self.w + y[i]*x[i, :]
                    self.mis += 1
            self.iter += 1
            self.testingMetrics(x)
            predictions.append(y_pred)
            if self.mis == 0:
                print("\nSuccess")
                print("Total iterations: " + str(self.iter))
                self.testingMetrics(x)
                break
            if self.iter > self.epochs:
                print("\nMisclassifications: " + str(self.mis))
                print("Total iterations: " + str(self.iter))
                self.testingMetrics(x)
                break
        return predictions

    def test(self, x, y):
        x = self.featureSelect(x)
        t = np.ones((np.size(x, 0), 1))
        x = np.append(x, t, axis=1)
        predictions = []
        for i in range(0, len(y)):
            y_pred = self.predict(np.dot(x[i, :], np.transpose(self.w)))
            predictions.append(y_pred)
        self.PerformanceMatrix(y, predictions)
        return np.array(predictions)


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

x, y = readfile('Dataset/a4a.txt')
inference = Perceptron(epochs=128)
y_pred_train = inference.train(x, y)
print("\nTesting...")
x_test, y_test = readfile('Dataset/a4a_t.txt')
y_pred_test = inference.test(x_test, y_test)
calculate_accuracy(y_test, y_pred_test)


# Multiclass Classification

# x, y = readfile('Dataset/iris.txt')
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     test_size=0.75, random_state=42
# )

# y_train_1 = np.array(y_train)
# y_train_2 = np.array(y_train)
# y_test_1 = np.array(y_test)
# y_test_2 = np.array(y_test)

# y_train_1[y_train_1 >= 2.0] = -1.0
# y_train_2[y_train_2 != 3.0] = -1.0
# y_train_2[y_train_2 == 3.0] = 1.0

# y_test_1[y_test_1 >= 2.0] = -1.0
# y_test_2[y_test_2 != 3.0] = -1.0
# y_test_2[y_test_2 == 3.0] = 1.0

# classifier1 = Perceptron(epochs=128)
# classifier2 = Perceptron(epochs=128)
# print("\nTraining classifier 1")
# y_pred_train_1 = classifier1.train(x_train, y_train_1)
# print("\nTraining classifier 2")
# y_pred_train_2 = classifier2.train(x_train, y_train_2)
# print("\nTesting...")
# print("\nTesting classifier 1")
# y_pred_test_1 = classifier1.test(x_test, y_test_1)
# print("\nTesting classifier 2")
# y_pred_test_2 = classifier2.test(x_test, y_test_2)

# y_pred = []
# for clf1, clf2 in zip(y_pred_test_1, y_pred_test_2):
#     if (clf1 == 1):
#         y_pred.append(1.0)
#     elif (clf2 == 1):
#         y_pred.append(3.0)
#     else:
#         y_pred.append(2.0)

# calculate_accuracy(y_test, y_pred)
