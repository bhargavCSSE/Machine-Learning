import pandas as pd
import numpy as np
import math

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
    x = pd.DataFrame(x).fillna(-1)
    x = x.assign(label=y)
    x = x.values.tolist()
    return x

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
        for val in values:
            question = Question(col, val)
            right, left = split_data(rows, question)

            if len(right) == 0 or len(left) == 0:
                continue
 
            gain = infoGain(right, left, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return leaf(rows)

    true_rows, false_rows = split_data(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return node(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, leaf):
        return node.y_pred

    if node.question.match(row):
        return classify(row, node.right_node)
    else:
        return classify(row, node.left_node)

def predict(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label] / total * 100)) + "%"
    return probs



training_data = readfile('Dataset/iris.txt')
print("Building the tree")
my_tree = build_tree(training_data)
print("Done")
testing_data = readfile('Dataset/iris.txt')

for row in testing_data:
    print("Actual: %s. Predicted: %s" %
          (row[-1], predict(classify(row, my_tree))))
