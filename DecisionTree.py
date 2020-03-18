import numpy as np
import math
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import confusion_matrix, accuracy_score


def best_split_point(X, y, column):
    # sorting y by the values of X makes
    # it almost trivial to count classes for
    # above and below any given candidate split point.
    ordering = np.argsort(X[:, column])
    classes = y[ordering]

    # these vectors tell us how many of each
    # class are present "below" (to the left)
    # of any given candidate split point.
    class_0_below = (classes == 0).cumsum()
    class_1_below = (classes == 1).cumsum()

    # Subtracting the cummulative sum from the total
    # gives us the reversed cummulative sum. These
    # are how many of each class are above (to the
    # right) of any given candidate split point.
    #
    # Because class_0_below is a cummulative sum
    # the last value in the array is the total sum.
    # That means we don't need to make another pass
    # through the array just to get the total; we can
    # just grab the last element.
    class_0_above = class_0_below[-1] - class_0_below
    class_1_above = class_1_below[-1] - class_1_below

    # below_total = class_0_below + class_1_below
    below_total = np.arange(1, len(y)+1)
    # above_total = class_0_above + class_1_above
    above_total = np.arange(len(y)-1, -1, -1)

    # we can now calculate Gini impurity in a single
    # vectorized operation. The naive formula would be:
    #
    #     (class_1_below/below_total)*(class_0_below/below_total)
    #
    # however, divisions are expensive and we can get this down
    # to only one division if we combine the denominator term.
    gini = class_1_below * class_0_below / (below_total ** 2) + class_1_above * class_0_above / (above_total ** 2)
    gini[np.isnan(gini)] = 1

    # we need to reverse the above sorting to
    # get the rule into the form C_n < split_value.
    best_split_rank = np.argmin(gini)
    best_split_gini = gini[best_split_rank]
    best_split_index = np.argwhere(ordering == best_split_rank).item(0)
    best_split_value = X[best_split_index, column]

    return best_split_gini, best_split_value, column


class Node:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_leaf = True
        self.column = None
        self.split_point = None
        self.children = None

    def is_pure(self):
        p = self.probabilities()
        if p[0] == 1 or p[1] == 1:
            return True
        return False

    def split(self, depth=0):
        X, y = self.X, self.y
        if self.is_leaf and not self.is_pure():
            splits = [best_split_point(X, y, column) for column in range(X.shape[1])]
            splits.sort()
            gini, split_point, column = splits[0]
            self.is_leaf = False
            self.column = column
            self.split_point = split_point

            below = X[:, column] <= split_point
            above = X[:, column] > split_point

            self.children = [
                Node(X[below], y[below]),
                Node(X[above], y[above])
            ]

            if depth:
                for child in self.children:
                    child.split(depth-1)

    def probabilities(self):
        return np.array([
            np.mean(self.y == 0),
            np.mean(self.y == 1),
        ])

    def predict_proba(self, row):
        if self.is_leaf:
            return self.probabilities()
        else:
            if row[self.column] <= self.split_point:
                return self.children[0].predict_proba(row)
            else:
                return self.children[1].predict_proba(row)


class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = int(max_depth)
        self.root = None

    def fit(self, X, y):
        self.root = Node(X, y)
        self.root.split(self.max_depth)

    def predict_proba(self, X):
        results = []
        for row in X:
            p = self.root.predict_proba(row)
            results += [p]
        return np.array(results)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


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


instance1 = decisionTree('Dataset/iris.txt')

X_in, y_in = instance1.readfile()
X = np.array(X_in)
y = y_in

# Data = load_iris()
# X = Data.data
# y = Data.target

# Data = load_breast_cancer()
# X = Data.data
# y = Data.target

model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)
y_hat = model.predict(X)
p_hat = model.predict_proba(X)[:, 1]
print(confusion_matrix(y, y_hat))
print('Accuracy:', accuracy_score(y, y_hat))
print(X.shape)
print(y.shape)
