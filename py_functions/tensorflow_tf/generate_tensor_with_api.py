import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

all_random = tf.random_normal((2, 2), 1.0, 1.0)


with tf.Session() as sess:

    all_zeros = tf.zeros((2, 2), dtype=tf.float32)
    print(all_zeros.eval())

    all_ones = tf.ones((2, 2), tf.float32)
    print(all_ones.eval())

    print(all_random.eval())

