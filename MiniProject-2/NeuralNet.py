import numpy as np


def calculate_accuracy(y_pred, y_true):
    mis = 0
    for pred, true in zip(y_pred, y_true):
        if pred != true:
            mis += 1
    return (len(y_pred) - mis)/len(y_pred)

class NeuralNetwork:
    def __init__(self, layers=[10], learning_rate = 10e-4):
        self.weights = []
        self.network = []
        self.layers = layers
        self.learning_rate = learning_rate
        np.random.seed(42)

        # Performance parameters
        self.train_err = []
        self.val_err = []
        self.test_err = []

    def sigmoid(self, x, deriv=False):
        if(deriv == True):
            return x*(1-x)

        return 1/(1+np.exp(-x))

    def step(self, x):
        x[x >= 0.5] = int(1)
        x[x < 0.5] = int(0)
        return x

    def softmax(self, x):
        val = np.exp(x)
        return val/val.sum(axis=1, keepdims=True)

    def generateModel(self, x, y):
        iH_size = self.layers[0]
        fH_size = self.layers[len(self.layers)-1]

        w_in = 2*np.random.random_sample((np.size(x, 1) + 1, iH_size)) - 1
        self.weights.append(w_in)

        for i in range(len(self.layers)-1):
            w = 2 * \
                np.random.random_sample(
                    (self.layers[i] + 1, self.layers[i+1])) - 1
            self.weights.append(w)

        w_out = 2*np.random.random_sample((fH_size + 1, np.size(y, 1))) - 1
        self.weights.append(w_out)

        #Generate Network
        total_layers = len(self.layers) + 2
        for i in range(total_layers):
            if(i == 0):
                self.network.append(np.hstack((np.ones((x.shape[0], 1)), x)))
            elif(i != total_layers-1):
                self.network.append(np.hstack((np.ones(
                    (self.network[i-1].shape[0], 1)), self.sigmoid(np.dot(self.network[i-1], self.weights[i-1])))))
            else:
                self.network.append(self.softmax(
                    np.dot(self.network[i-1], self.weights[i-1])))

    def train(self, x, y, epochs, print_error=True):
        self.generateModel(x, y)
        total_layers = len(self.network)

        for itr in range(epochs):

            #Feed Forward
            for i in range(total_layers):
                if(i == 0):
                    self.network[0] = np.hstack((np.ones((x.shape[0], 1)), x))
                elif(i != total_layers-1):
                    self.network[i] = np.hstack((np.ones(
                        (self.network[i-1].shape[0], 1)), self.sigmoid(np.dot(self.network[i-1], self.weights[i-1]))))
                else:
                    self.network[i] = self.softmax(
                        np.dot(self.network[i-1], self.weights[i-1]))

            dw = []
            #Backpropagation
            output_error = y - self.network[total_layers-1]

            for layer_count in range(total_layers-1, 0, -1):
                if(layer_count == total_layers-1):
                    delta = output_error * self.sigmoid(self.network[layer_count])
                    output_error = delta.dot(self.weights[layer_count-1].T)
                    dw.append(self.learning_rate * self.network[layer_count-1].T.dot(delta))
                else:
                    delta = output_error[:, 1:]*self.sigmoid(self.network[layer_count][:, 1:], deriv=True)
                    output_error = delta.dot(self.weights[layer_count-1].T)
                    dw.append(self.learning_rate * self.network[layer_count-1].T.dot(delta))

            # Update weights
            for layer_count in range(total_layers-1, 0, -1):
                self.weights[layer_count-1] += dw[total_layers-1 - layer_count]

            if(itr % 1 == 0):
                train_err = -np.sum(y * np.log(self.network[total_layers-1]))
                self.train_err.append(train_err)
                if(print_error):
                    print("Iteration: {}\tTraining error: {}\t".format(itr, train_err))

        print("Training finished")

    def test(self, x):
        total_layers = len(self.network)
        for i in range(total_layers):
            if(i == 0):
                self.network[0] = np.hstack((np.ones((x.shape[0], 1)), x))
            elif(i != total_layers-1):
                self.network[i] = np.hstack((np.ones(
                    (self.network[i-1].shape[0], 1)), self.sigmoid(np.dot(self.network[i-1], self.weights[i-1]))))
            else:
                self.network[i] = self.softmax(
                    np.dot(self.network[i-1], self.weights[i-1]))

        output = []
        for elem in self.network[total_layers-1]:
            output.append(np.argmax(elem))
        return output

    def train_t(self, x, y, x_val, y_val, x_test, y_test, epochs, print_error=True):
        self.generateModel(x, y)
        total_layers = len(self.network)

        for itr in range(epochs):

            #Feed Forward
            for i in range(total_layers):
                if(i == 0):
                    self.network[0] = np.hstack((np.ones((x.shape[0], 1)), x))
                elif(i != total_layers-1):
                    self.network[i] = np.hstack((np.ones(
                        (self.network[i-1].shape[0], 1)), self.sigmoid(np.dot(self.network[i-1], self.weights[i-1]))))
                else:
                    self.network[i] = self.softmax(np.dot(self.network[i-1], self.weights[i-1]))

            dw = []
            #Backpropagation
            output_error = y - self.network[total_layers-1]

            for layer_count in range(total_layers-1, 0, -1):
                if(layer_count == total_layers-1):
                    delta = output_error * self.sigmoid(self.network[layer_count])
                    output_error = delta.dot(self.weights[layer_count-1].T)
                    dw.append(self.learning_rate * self.network[layer_count-1].T.dot(delta))
                else:
                    delta = output_error[:, 1:]*self.sigmoid(self.network[layer_count][:, 1:], deriv=True)
                    output_error = delta.dot(self.weights[layer_count-1].T)
                    dw.append(self.learning_rate * self.network[layer_count-1].T.dot(delta))

            # Update weights
            for layer_count in range(total_layers-1, 0, -1):
                self.weights[layer_count-1] += dw[total_layers-1 - layer_count]

            if(itr % 1 == 0):
                train_err = -np.sum(y * np.log(self.network[total_layers-1]))
                self.train_err.append(train_err)
                val_err = self.test_t(x_val, y_val)
                self.val_err.append(val_err)
                test_err = self.test_t(x_test, y_test)
                self.test_err.append(test_err)
                if(print_error):
                    print("Iteration: {}\tTraining error: {}\tValidation error: {}\tTesting error {}".format(itr, train_err, val_err, test_err))

        print("Training finished")

    def test_t(self, x, y):
        total_layers = len(self.network)
        for i in range(total_layers):
            if(i == 0):
                self.network[0] = np.hstack((np.ones((x.shape[0], 1)), x))
            elif(i != total_layers-1):
                self.network[i] = np.hstack((np.ones(
                    (self.network[i-1].shape[0], 1)), self.sigmoid(np.dot(self.network[i-1], self.weights[i-1]))))
            else:
                self.network[i] = self.softmax(
                    np.dot(self.network[i-1], self.weights[i-1]))

        test_err = -np.sum(y * np.log(self.network[total_layers-1]))
        return test_err

        
