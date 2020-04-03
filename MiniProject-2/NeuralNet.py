import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers=[10]):
        self.weights = []
        self.network = []
        self.layers = layers
        np.random.seed(1)

    def readfile(self, filename):
        data = open(filename)
        x = []
        y = []
        for index, line in enumerate(data):
            line = line.split(None)
            temp_x = []  # i love you honeyboo
            temp_y = []
            for i in range(len(line)):
                if(i == (len(line)-1)):
                    temp_y.append(float(line[i]))
                else:
                    temp_x.append(float(line[i]))
            y.append(temp_y)
            x.append(temp_x)
        return np.array(x), np.array(y)

    def nonlin(self, x, deriv=False):
        if(deriv == True):
            return x*(1-x)

        return 1/(1+np.exp(-x))

    def step(self, x):
        x[x >= 0.5] = int(1)
        x[x < 0.5] = int(0)
        return x

    def generateModel(self, x, y):
        iH_size = self.layers[0]
        fH_size = self.layers[len(self.layers)-1]
        w_in = 2*np.random.random_sample((np.size(x, 1), iH_size)) - 1
        self.weights.append(w_in)
        for i in range(len(self.layers)-1):
            w = 2 * \
                np.random.random_sample((self.layers[i], self.layers[i+1])) - 1
            self.weights.append(w)
        w_out = 2*np.random.random_sample((fH_size, np.size(y, 1))) - 1
        self.weights.append(w_out)

        #Generate Network
        total_layers = len(self.layers) + 2
        for i in range(total_layers):
            if(i == 0):
                self.network.append(x)
            else:
                self.network.append(self.nonlin(
                np.dot(self.network[i-1], self.weights[i-1])))

    def train(self, x, y, epochs):
        self.generateModel(x, y)
        total_layers = len(self.network)

        for i in range(epochs):

            #Update network
            for i in range(total_layers):
                if(i == 0):
                    self.network[0] = x
                else:
                    self.network[i] = self.nonlin(np.dot(self.network[i-1], self.weights[i-1]))

            #Backpropagation
            output_error = y - self.network[total_layers-1]
            if(epochs % 1000 == 0):
                 print("Training error: {}".format(np.mean(np.abs(output_error))))

            for layer_count in range(total_layers-1, 0, -1):
                delta = output_error*self.nonlin(self.network[layer_count], deriv=True)
                output_error = delta.dot(self.weights[layer_count-1].T)
                self.weights[layer_count -1] += self.network[layer_count-1].T.dot(delta)

    def test(self, x, step=False):
        total_layers = len(self.network)
        for i in range(total_layers):
            if(i == 0):
                self.network[0] = x
            else:
                self.network[i] = self.nonlin(
                np.dot(self.network[i-1], self.weights[i-1]))

        output = self.network[total_layers-1]
        if(step == True):
            output = self.step(output)
        print(output)
        return output

nn = NeuralNetwork(layers=[3, 5])
x, y = nn.readfile("data.txt")
nn.train(x, y, 10000)
y = nn.test(x)
