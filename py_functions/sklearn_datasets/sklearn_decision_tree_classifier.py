import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = \
    train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42, test_size=0.25)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)

accuracy_train = tree.score(x_train, y_train)
print(accuracy_train)
accuracy_test = tree.score(x_test, y_test)
print(accuracy_test)

