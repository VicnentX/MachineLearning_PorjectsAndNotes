import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array(X[:, 0]**2 + X[:,1] < 1.5, dtype='int')

# 相当于加了噪音
for _ in range(20):
    y[np.random.randint(200)] = 1

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print(log_reg)
print("accuracy with train data")
print(log_reg.score(X_train, y_train))
print("accuracy with test data")
print(log_reg.score(X_test, y_test))


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(log_reg, axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()


"""
下面用polynomial logistic regression 
"""


def PolynomialLogisticRegression(degree):
    return Pipeline([
        # 管道第一步：给样本特征添加多形式项；
        ('poly', PolynomialFeatures(degree=degree)),
        # 管道第二步：数据归一化处理；
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])



# 过拟合了 ....so c = 0.1, degree=20 panelty = c1 会缓解
poly_log_reg = PolynomialLogisticRegression(degree=20)
poly_log_reg.fit(X_train, y_train)

print(poly_log_reg)
print("accuracy with train data")
print(poly_log_reg.score(X_train, y_train))
print("accuracy with test data")
print(poly_log_reg.score(X_test, y_test))

plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()


# """
#
# 　2）使用逻辑回归算法（不添加多项式项）
#
# """
#
# log_reg = LogisticRegression()
# log_reg.fit(X, y)
#

#
#
# plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
# plt.scatter(X[y == 0, 0], X[y == 0, 1])
# plt.scatter(X[y == 1, 0], X[y == 1, 1])
# plt.show()



