import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# build a graph
a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

# launch the graph in a session
with tf.Session() as sess:
    output = sess.run(c)
    print(output)



print("---example 2-----")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a * b

with tf.Session() as sess:
    print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
    print(type(adder_node))

print("-----example 3 linear model ------")

# model parameters
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# inputs and outputs
x = tf.placeholder(tf.float32)

linear_model = w * x + b

y = tf.placeholder(tf.float32)

# loss
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

# optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print(sess.run([w, b]))
