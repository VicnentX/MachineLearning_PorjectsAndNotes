# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[1])))
X_test = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[1])))

# One thing I briefly want to mention is that is
# the default optimization algorithm parameter was solver = liblinear
# and it took 2893.1 seconds to run with a accuracy of 91.45%
# When I set solver = lbfgs , it took 52.86 seconds to run with an accuracy of 91.3%.
# Changing the solver had a minor effect on accuracy, but at least it was a lot faster.

# this step is not affect the result
X_train, X_test = X_train / 255, X_test / 255
#
logistic_regression_model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=100, verbose=2)
logistic_regression_model.fit(X_train, y_train)
# score
y_test_hat = logistic_regression_model.predict(X_test)
accuracy = logistic_regression_model.score(X_test, y_test)
print("accuracy ： ", accuracy)

# show confuse matrix
print("-------------------")
print(f"y_test shape is {y_test.shape}")
print("-------------------")
cm = confusion_matrix(y_test, y_test_hat)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = f"MNIST's Accuracy Score: {accuracy}"
plt.title(all_sample_title, size=15)
plt.show()
