import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

plt = tf.placeholder(tf.float32, [None, 2])
print(plt)
print(plt.graph)
print(plt.shape)
# 当我堵过来的时候 形状是不固定的 之后我set shape就是要改变形状
plt.set_shape((3, 2))
print(plt)
print(plt.graph)
print(plt.shape)
# 对于静态形状来说 一旦形状固定了 不能再设置静态形状 + 不能跨维度修改
# 动态形状可以创建一个新的张亮 形状不同


newT = tf.reshape(plt, [6, 1])
print(newT)
print(newT.graph)
print(newT.shape)
print("我理解了 这边会发现newT和plt的地址是一样的 因为其实指的是一个东西 只是这个东西有两个shape 一个静态 一个动态")

with tf.Session() as sess:
    pass

print("eg2----------------------------------")
x = tf.placeholder(tf.int32, shape=[4])
y, _ = tf.unique(x)
z = tf.shape(y)
with tf.Session() as sess:
    print(sess.run(z, feed_dict={x: [1,2,3,4]}))
    print(sess.run(z, feed_dict={x: [1,2,3,2]}))


print("eg3 ---------------------------------")
# -*- coding:utf-8 -*-

# 定义一个第一维度不确定的tensor
a = tf.placeholder(tf.float32, [None, 128])
b = tf.placeholder(tf.int32, [2, 5, 6])
c = tf.placeholder(tf.int32, [4])

# 查询张量a的静态维度
static_shape_a = a.shape.as_list()
static_shape_b = b.shape.as_list()

print(static_shape_a, static_shape_b)

# 设置张量a的静态维度,a必须有最少一个维度上是不确定才可以设置静态维度
a.set_shape([32, 128])
# b声明的时候3个静态维度已经是确定的了，再设置b的静态维度会报错
# b.set_shape([4,5,3])

# 查询张量a设置静态维度后的静态维度
static_shape_a = a.shape.as_list()
print(static_shape_a)
print(c.shape)
print(type(c.shape))


print("eg4--------------------------------------")
# 定义一个第一维度不确定的tensor
a = tf.placeholder(tf.float32, [None, 128])

# 查询a的动态维度
dynamic_shape = tf.shape(a)
print(dynamic_shape)

# 改变a的动态维度
a = tf.reshape(a, [32, 64, 2])

# 查询a的动态维度
dynamic_shape = tf.shape(a)
print(dynamic_shape)
