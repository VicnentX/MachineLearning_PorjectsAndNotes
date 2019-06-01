import tensorflow as tf

# 变量可以持久化保存
# 定义变量op的时候 一定要在会话当中去运行初始化

a = tf.constant([1,2,3,4,5])
var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0))
# 此时没有初始化var
print(a)
print(var)
# 定义了初始化 但是没有run 需要在sess里面run
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run([a, var]))



