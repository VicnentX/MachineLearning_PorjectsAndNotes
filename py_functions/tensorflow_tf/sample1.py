import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# new a graph and in this graph new some tensor
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11.0)   # 这个a与下面那个a没有关系
    print(c.graph)

f = tf.Graph()
print(f)
with f.as_default():
    d = tf.constant(11.0)   # 这个a与下面那个a没有关系
    print(d.graph)


print("_______________________")
# implement a add operation

a = tf.constant(5.0)
b = tf.constant(6.0)
print(a)
print(b)
sum1 = tf.add(a, b)

# 默认的这张图 相当于给程序分配一段内存
graph = tf.get_default_graph()
print(graph)

# 会话session 是运行图的结构 分配资源计算 掌握资源（变量 队列 （多）线程）
# 一次只能运行一个图 （不指定的情况下）只能运行默认的图 tf.get_default_graph()
with tf.Session() as sess:  #这个时候不能运行g里面的a
    print(sess.run(sum1))
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)

# 指定某一张图运行
# with 是上下文管理器 结束之后会自动帮我close（）
with tf.Session(graph=g) as sess:
    print("# run 只能run图里面的变量")
    print(sess.run(c))
    print("# 运行图是可以的 只要我在外层定义好或者在别的图里面定义好了 但是不能run 因为不在图里面")
    print(d.graph)
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)

# config 显示运行的设备是什么
print("# config 显示运行的设备是什么")
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print("# run 只能run图里面的变量")
    print(sess.run(sum1))
    print(sum1.eval())
    print("# 运行图是可以的 只要我在外层定义好或者在别的图里面定义好了 但是不能run 因为不在图里面")
    print(d.graph)
    print(a.graph)
    print(sum1.graph)
    print(sess.graph)
    print(sess.run([sum1, a, b]))
    result = sess.run([sum1, a, b])
    print(type(result))


print("# 不是op 不能运行 ， 但是可以重载")
var1 = 2.0
sum2 = var1 + a
# 会默认吧sum2转换成op类型
print("sum2 type ------------", sum2)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum2))


print("what is 训练模型? 实时提供数据去进行训练------placeholder是一个占位符，feed_dict一个字典")
plt = tf.placeholder(tf.float32, (2, 3))
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(plt, feed_dict={plt:[[1,2,3],[4,5,6]]}))

print("比如说我只知道feature数量 不知道样本数")
plt2 = tf.placeholder(tf.float32, (None, 3))
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(plt2, feed_dict={plt2:[[1,2,3],[4,5,6],[7,8,9]]}))

