import numpy as np
import math
import pandas as pd


def entropyFunction(p, tp):
    e = -(p/tp)*math.log(p/tp)
    return e


def calculateEntropy(count1, count2):
    if (count1 == 0 or count2 == 0):
        return 0
    totalCount = count1 + count2
    return entropyFunction(count1, totalCount) + entropyFunction(count2, totalCount)


def entropyPerSplit(split):
    s = 0
    n = len(split)
    classes = set(split)
    for c in classes:
        n_c = sum(split == c)
        e = n_c*1.0/n * calculateEntropy(sum(split == c), sum(split != c))
        s += e
    return s, n


def entropy(y_predict, y_real):
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropyPerSplit(y_real[y_predict])
    s_false, n_false = entropyPerSplit(y_real[~y_predict])
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false
    return s


class decisionTree:
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
        return x, y


instance1 = decisionTree('Dataset/a4a.txt')
# y, x = instance1.svm_read_problem()
# print(x)
x, y = instance1.readfile()
print(x)
print(y.shape)
