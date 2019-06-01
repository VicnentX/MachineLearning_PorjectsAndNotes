import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

X = iris.data
y = iris.target

# 我们指做binary classification

X = X[y < 2, :2]
y = y[y < 2]

print(X)
print(y)
print(X.shape)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")
plt.show()


print("-------knn classification--------")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
accuracy_knn = knn_clf.score(X_test, y_test)
print(accuracy_knn)
print(knn_clf)
# knn 算法的决策边界是一条曲线（作用在两个类别



