import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class leaf:
    def __init__(self, data):
        self.y_pred = count_classes(data)


class node:
    def __init__(self, question, right_node, left_node):
        self.question = question
        self.right_node = right_node
        self.left_node = left_node


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value


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
    return x, y

def count_classes(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

def split_data(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def entropy(rows):
    counts = count_classes(rows)
    entropy = 0
    for classes in counts:
        prob = counts[classes] / float(len(rows))
        entropy += -prob*math.log(prob)
    return entropy

def infoGain(right, left, current_uncertainty):
    p = 1.0*(len(right)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(right) - (1 - p) * entropy(left)

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for elem in values:
            question = Question(col, elem)
            right, left = split_data(rows, question)
            gain = infoGain(right, left, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    
    return best_gain, best_question

def grow_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return leaf(rows)

    true_rows, false_rows = split_data(rows, question)

    true_branch = grow_tree(true_rows)
    false_branch = grow_tree(false_rows)

    return node(question, true_branch, false_branch)

def predict(row, tree):
    if isinstance(tree, leaf):
        return tree.y_pred

    if tree.question.match(row):
        return predict(row, tree.right_node)
    else:
        return predict(row, tree.left_node)

def output_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs


def PerformanceMatrix(y_actual, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for test_instance_result, label in zip(y_pred, y_actual):
        if ((test_instance_result > 0.5) and (label > 0.5)):
            tp += 1
        if ((test_instance_result <= 0.5) and (label <= 0.5)):
            tn += 1
        if ((test_instance_result > 0.5) and (label <= 0.5)):
            fp += 1
        if ((test_instance_result <= 0.5) and (label > 0.5)):
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn + 0.00001)
    precision = tp / (tp + fp + 0.00001)
    f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
    print("Accuracy:     ", accuracy)
    print("Recall:       ", recall)
    print("Precision:    ", precision)
    print("F1:           ", f1)

def featureSelect(X, x):
    mask = list(X.columns.values)
    x = x.loc[:, mask]
    return x

x, y = readfile('Dataset/iris.txt')
x = pd.DataFrame(x).fillna(-1)
x = x.assign(label=y)
x = x.values.tolist()

training_data, testing_data, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# train_data, y_train = readfile('Dataset/a4a.txt')
# train_data = pd.DataFrame(train_data).fillna(-1)
# train_data = train_data.assign(label=y_train)
# training_data = train_data.values.tolist()

print("Building the tree")
my_tree = grow_tree(training_data)
print("Done")

# testing_data, y_test = readfile('Dataset/a4a_t.txt')
# testing_data = pd.DataFrame(testing_data).fillna(-1)
# testing_data = featureSelect(train_data, testing_data)
# testing_data = testing_data.assign(label=y_test)
# testing_data = testing_data.values.tolist()

y_pred = []
for row in testing_data:
    y = (output_leaf(predict(row, my_tree)))
    for key, value in y.items():
        y_pred.append(float(key))
y_pred = np.array(y_pred)
PerformanceMatrix(y_test, y_pred)
