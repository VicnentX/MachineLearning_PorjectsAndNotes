import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

mnist = fetch_mldata("MNIST original")
# print(mnist)

x, y = mnist["data"], mnist["target"]
print("x.shape", x.shape)
print("x type", type(x))
print("element type of x", np.dtype(x[0, 0]))
print("y.shape", y.shape)

x_train = np.array(x[:60000], dtype=float)
y_train = np.array(y[:60000], dtype=float)
x_test = np.array(x[60000:], dtype=float)
y_test = np.array(y[60000:], dtype=float)



# no reduction
# print("x_train shape: ", x_train.shape)
# print("y_train shape: ", y_train.shape)
#
# knn_clf = KNeighborsClassifier()
# print(knn_clf.fit(x_train, y_train))
#
# rate = knn_clf.score(x_test, y_test)
# print("yield is : ")


# reduction
pca = PCA(0.8)  # 我希望成功率到80%就行了 她会决定k是多少
pca.fit(x_train)
x_train_reduction = pca.transform(x_train)
print("x_train_reduction shape is", x_train_reduction.shape)
x_test_reduction = pca.transform(x_test)
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_reduction, y_train)
rate = knn_clf.score(x_test_reduction, y_test)
print("yield is :", rate)
