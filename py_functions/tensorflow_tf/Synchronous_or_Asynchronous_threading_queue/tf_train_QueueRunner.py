import tensorflow as tf

"""
异步操作
子线程一直放数据，1， 2， 3， 4。。。这样放
queue.size().eval() = 1000
主线程 看到有数据就取了

"""

# 1 定义队列
Q = tf.FIFOQueue(1000, tf.float32)

# 2 子线程做的事情 循环 值+1 放入队列中
var = tf.Variable(0.0)  # var是tf.variable op
# 实现自增
data = tf.assign_add(var, tf.constant(1.0))
# 入队
en_q = Q.enqueue(data)


# 3 定义队列管理器op 指定多少子线程 指定子线程干什么
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)    # [en_q, op1, op2] * 2 这里【】里面三个op谁先谁后没法控制

# !!!!初始化op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化变量 run init_op
    sess.run(init_op)

    # 开启线程管理器
    coord = tf.train.Coordinator()

    # 真正 run 子线程
    threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程不断读取数据
    for i in range(300):
        print(sess.run(Q.dequeue()))
        print(f"--{i}")

# 到这里的时候 主线程结束 意味着session关闭 意味着资源释放 这个时候子线程还在工作 什么都不写会报错！！！！
    # 我此时回收子线程（调整 "# 真正 run 子线程" 行 里面加入线程管理员）
    coord.request_stop()
    coord.join(threads)
