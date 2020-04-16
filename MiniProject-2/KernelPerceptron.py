import numpy as np


class kernel_perceptron(object):
    def __init__(self, T=1):
        self.T = T

    def kernel(self, x1, x2):
        return (1 + np.dot(x1, x2))**2

    def generate_model(self, x, y):
        self.alpha = np.zeros(x.shape[0])
        self.mis = []
        self.itr = 0

    def sign(self, x):
        if x >= 0.0:
            return 1.0
        elif x < 0.0:
            return -1.0
        else:
            return 0.0

    def confusion_matrix(self, y_actual, y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for pred, label in zip(y_pred, y_actual):
            if ((pred > 0.0) and (label > 0.0)):
                tp += 1
            if ((pred <= 0.0) and (label <= 0.0)):
                tn += 1
            if ((pred > 0.0) and (label <= 0.0)):
                fp += 1
            if ((pred <= 0.0) and (label > 0.0)):
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn + 1e-5)
        precision = tp / (tp + fp + 1e-5)
        f1 = 2 * (precision * recall) / (precision + recall)

        return accuracy, recall, precision, f1

    def train(self, x, y):
        self.generate_model(x, y)
        n, D = x.shape

        # Kernel Matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(x[i], x[j])

        for epoch in range(self.T):
            y_hat = []
            mis = 0
            for i in range(n):
                # Predict
                y_hat.append(self.sign(np.sum(K[i, :]*self.alpha)))

                # Update alpha
                if y_hat[i] != y[i]:
                    self.alpha[i] += y[i]
                    mis += 1

            self.mis.append(mis)
            if (mis == 0):
                print('Success! with {} iterations'.format(epoch))
                break

        self.itr = epoch
        return self.confusion_matrix(y, y_hat)

    def test(self, x):
        n, D = x.shape
        y_hat = []

        # Kernel Matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(x[i], x[j])

        for i in range(n):
            # Predict
            y_hat.append(self.sign(np.sum(K[i, :]*self.alpha)))

        return y_hat
