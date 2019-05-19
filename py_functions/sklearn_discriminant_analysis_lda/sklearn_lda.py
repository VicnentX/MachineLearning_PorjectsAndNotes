"""
predict_proba返回的是一个 n 行 k 列的数组，
第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 3])
lda_model = LinearDiscriminantAnalysis(n_components=1)
lda_model.fit(X, y)

y_test_hat = lda_model.predict([[-0.8, -1], [1, 1.5], [0.6, 0.1]])
print(y_test_hat)

accuracy = lda_model.score([[-0.8, -1], [1, 1.5], [0.6, 0.1]], [1, 2, 2])
print(accuracy)

y_test_prob = lda_model.predict_proba([[-0.8, -1], [1, 1.5], [0.6, 0.1]])
print(y_test_prob)
