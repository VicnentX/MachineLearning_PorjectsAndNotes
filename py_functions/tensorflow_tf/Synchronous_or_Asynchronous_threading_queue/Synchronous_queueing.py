import tensorflow as tf

"""
# 模拟一下同步处理数据 然后才能取数据训练
"""

# 1。 首先定义队列
queue = tf.FIFOQueue(3, tf.float32)
    # 放数据
enq_many = queue.enqueue_many([[0.1, 0.2, 0.3],])

# 2。定义一些读取数据，读取数据过程 取数据+1 再放回队列
out_q = queue.dequeue()
data = out_q + 1
en_q = queue.enqueue(data)

# 3。之后在session里面运行
with tf.Session() as sess:
    # 舒适化队列:
    sess.run(enq_many)
    """
    注意 所有的run 都是run的op
    """

    # 处理数据：
    for i in range(100):
        # tensorflow 操作有""""""依赖性""""""，所以执行最后那个的话 前面也会自动执行 13 - 15 行，但是13行与10行无关
        sess.run(en_q)
    # 训练模型
    # 注意queue。size() 也是op
    for i in range(queue.size().eval()):
        print(sess.run(queue.dequeue()))






