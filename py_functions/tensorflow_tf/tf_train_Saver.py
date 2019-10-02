# https://blog.csdn.net/mzpmzk/article/details/78647699

import tensorflow as tf
import os
import numpy as np

def myregression():
    """
    自实现一个线性回归预测
    :return:
    """
    # 1. 准备数据， 1 个特征 【100， 1】 y【100】 100个sample
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

    # 矩阵相乘必须是二维的
    y_true = tf.matmul(x, [[0.7]]) + 0.8
    # y_true = x * 0.7 + 0.8


    # 2. 建立线性回归模型 1个特征 一个权重 一个偏执 y = x w + b
    # 随机给 w b 开始的时候 让她去计算损失 然后在当前状态下优化
    # ！！！！！！！！！用变量定义 才能优化
    # trainable 默认是 True 改成False的话 这个值就不能被改变了
    weight = tf.Variable(tf.random_normal((1,1), mean=0.0, stddev=1.0), name="w", trainable=True)
    bias = tf.Variable(0.0, name="b")
    y_predict = tf.matmul(x, weight) + bias

    # 3. 建立损失函数， 均芳误差
    error = tf.square(y_predict - y_true)
    loss = tf.reduce_mean(error)

    # 4. 梯度下降 优化损失
    # 学习旅 0-  1 之间取
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 5. 因为有variable 所以要初始化变量的op
    init_op = tf.global_variables_initializer()

    '''
    # 收集tensor，收集变量 收集卸载session前面
    # 下面收集loss 因为他只是一个数字 所以就是scalar
    # 之后就合并变量，写入事件簿
    '''
    tf.summary.scalar("losses", loss)   #losses 是在后台显示的名字
    tf.summary.histogram("weights", weight)     #weight 是二维的 所以用histogram
    # 合并变量写入事件步，定义合并tensor的op
    merged = tf.summary.merge_all()


    """
    定义一个保存模型的实例
    """
    saver = tf.train.Saver()

    # open session
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 打印随机最初的的w b
        print(f"初时的w : {weight.eval()} , b : {bias.eval()}")


        # 建立事件文件
        writer = tf.summary.FileWriter("/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/", graph=sess.graph)


        """
        这边当我重新训练 我就从之前saver的时候开始训练：
        所以我要加载模型，覆盖模型当中随机定义的参数，从上次训练的参数结果开始
        """
        if os.path.exists("/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/checkpoint/checkpoint"):
            saver.restore(sess, "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/checkpoint/model_for_w&b")


        # 这里开始循环优化：
        for i in range(1000):

            # !!!!!运行 一次 优化 （特别注意这里run指运行一次优化）
            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)

            # 每次的数据都要写入记事簿
            writer.add_summary(summary, i)  #每次写入第i个值

            # 打印优化后的的w b
            print(f"第 {i + 1} 优化后的w : {weight.eval()} , b : {bias.eval()}")

        saver.save(sess, "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/checkpoint/model_for_w&b")
        # 这次我的数据文件 model_for_w&b.data-00000-of-00001

    return None


if __name__ == "__main__":
    myregression()