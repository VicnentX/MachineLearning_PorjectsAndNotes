import tensorflow as tf

with tf.Session() as sess:
    x = tf.cast([[1,1,1], [1,1,1]], tf.int32)
    print(tf.reduce_sum(x).eval())
    print(tf.reduce_sum(x, 0).eval())
    print(tf.reduce_sum(x, 1).eval())
    print(tf.reduce_sum(x, 1, keepdims=True).eval())
    print(tf.reduce_sum(x, [0, 1]).eval())
