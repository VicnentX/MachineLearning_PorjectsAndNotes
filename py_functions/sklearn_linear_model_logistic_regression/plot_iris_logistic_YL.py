#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Logistic Regression 3-class Classifier
=========================================================


"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from functionToCall import plot_confusion_matrix222

# load the data
iris = datasets.load_iris()


logreg = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=1000)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    stratify=iris.target, random_state=42, test_size=0.25)

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X_train, y_train)


y_test_hat = logreg.predict(X_test)
num_correct = 0
for i in range(len(y_test)):
    if y_test_hat[i]==y_test[i]:
        num_correct +=1
        
Accuracy_rate = num_correct/len(y_test)
print("Accuracy Rate = ", Accuracy_rate)

confusion_matrix(y_test, y_test_hat, labels=[0, 1, 2])
class_names = iris.target_names
plot_confusion_matrix222(y_test, y_test_hat, classes=class_names,
                      title='Confusion matrix, without normalization')

plot_confusion_matrix222(y_test, y_test_hat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()



