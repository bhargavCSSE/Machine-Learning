import numpy as np
# import math
import pandas as pd
scipy = 0
sparse = 0


class decisionTree:
    def __init__(self, filename):
        self.filename = filename

    def readfile(self):
        data = open(self.filename)
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
                temp_x[int(name)] = float(value)
            x = x + [temp_x]
        return x, y


instance1 = decisionTree('Dataset/iris.txt')
# y, x = instance1.svm_read_problem()
# print(x)
x, y = instance1.readfile()
print(x)
print(y)
