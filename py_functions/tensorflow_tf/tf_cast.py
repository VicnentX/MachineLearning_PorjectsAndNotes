import tensorflow as tf

a = tf.cast([[1,2,3],[4,5,6]], tf.float32)
with tf.Session() as sess:
    print(a.eval())
    print(a.dtype)
