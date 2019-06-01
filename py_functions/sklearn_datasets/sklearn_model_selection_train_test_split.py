"""
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

import numpy as np
from sklearn.model_selection import train_test_split
x, y = np.arange(10).reshape((5, 2)), np.array(range(5))
print("x :")
print(x)
print("y :")
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train)
print(y_train)
print(x_test)
print(y_test)