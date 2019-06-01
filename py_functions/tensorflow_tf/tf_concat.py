import tensorflow as tf

with tf.Session() as sess:
    a = [[1,2,3], [4,5,6]]
    b = [[7,8,9], [1,2,3]]
    c = tf.concat([a,b], axis=0)
    d = tf.concat([a,b], axis=1)
    print(c.eval())
    print(d.eval())

