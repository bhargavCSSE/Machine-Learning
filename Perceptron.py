import numpy as np
import pandas as pd


class Perceptron(object):
    def __init__(self, filename):
        self.filename = filename

    def readfile(self):
        data = open(self.filename)
        y = np.array([])
        x = []
        for index, line in enumerate(data):
            line = line.split(None, 1)
            if len(line) == 1:
                line += ['']
            label, features = line
            y = np.append(y, float(label))
            temp_x = {}
            for elem in features.split(None):
                name, value = elem.split(':')
                temp_x[int(name)] = (float(value))
            x = x + [temp_x]
        x = pd.DataFrame(x).fillna(0)
        x = x.to_numpy()
        return x, y


inference = Perceptron('Dataset/iris.txt')
x, y = inference.readfile()
print(x)
print(y)
print(x.shape)
print(y.shape)
