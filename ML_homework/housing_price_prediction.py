import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

m, n = housing.data.shape

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
#target price
t = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="t")
XT = tf.transpose(X)
w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), t)

# predicted price
y = tf.matmul(X, w)
MSE = tf.div(tf.matmul(tf.transpose(y - t), y - t), m)

with tf.Session() as sess:
    X_value = X.eval()
    t_value = t.eval()
    XT_value = XT.eval()
    w_value = w.eval()
    y_value = y.eval()
    MSE_value = MSE.eval()


