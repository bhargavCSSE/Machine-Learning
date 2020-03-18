import numpy as np
import pandas as pd


class Perceptron(object):
    def __init__(self, epochs):
        self.epochs = epochs
        self.w = np.array([])
        self.iter = 0
        self.mis = 0
        self.x = pd.DataFrame()
        self.y = np.array([])

    def readfile(self, filename):
        data = open(filename)
        y = []
        x = []
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

    def Initialize(self, filename):
        x, y = self.readfile(filename)
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

    def testingMetrics(self, x, printTest=False):
        if printTest is True:
            for i in range(0, np.size(x, 0)):
                y_pred = self.predict(np.dot(x[i, :], np.transpose(self.w)))
                print(str(x[i]) + " " + str(y_pred))
        total = np.size(x, 0)
        accuracy = (total-self.mis)/total
        print("Training accuracy: " + str(accuracy))

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

    def train(self, filename):
        x, y = self.Initialize(filename)
        while(True):
            self.mis = 0
            for i in range(0, len(y)):
                y_pred = self.predict(np.dot(x[i, :], np.transpose(self.w)))
                if (y[i]*y_pred) <= 0:
                    self.w = self.w + y[i]*x[i, :]
                    self.mis += 1
            self.iter += 1
            if self.mis == 0:
                print("Success")
                print("Total iterations: " + str(self.iter))
                self.testingMetrics(x)
                break
            if self.iter > self.epochs:
                print("Misclassifications: " + str(self.mis))
                self.testingMetrics(x)
                break
        return 0

    def featureSelect(self, x):
        mask = list(self.x.columns.values)
        x = x.loc[:, mask]
        x = x.to_numpy()
        return x

    def test(self, filename):
        x, y = self.readfile(filename)
        x = self.featureSelect(x)
        t = np.ones((np.size(x, 0), 1))
        x = np.append(x, t, axis=1)
        predictions = []
        for i in range(0, len(y)):
            y_pred = self.predict(np.dot(x[i, :], np.transpose(self.w)))
            predictions.append(y_pred)
        print("Performance metrics:")
        self.PerformanceMatrix(x, y, predictions)
        return np.array(predictions)


inference = Perceptron(epochs=1)
inference.train('Dataset/a4a.txt')
pred = inference.test('Dataset/a4a_t.txt')
