import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 定义初始化权重的函数
def weight_variable(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义初始化偏执的函数
def biad_variable(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model_DIY():
    """
    自定义卷积模型
    :return:
    """
    # 1。 准备数据的占位符 X[Node, 784] y_true [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    #2. the first conv layer(conv + activate + pool)
    with tf.variable_scope("conv1"):
        # 用32个filter 5*5*1
        w_conv1 = weight_variable([5,5,1,32])
        b_conv1 = biad_variable([32])

        # 对x进行形状改变
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # 从[None,28,28,1] -> [None,28,28,32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1)

        #pooling 2*2 ,stride=2, [None,28,28,32] ->[None,14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    # 3. the first conv layer(conv + activate + pool) 5*5*32, 64个filter， stride=1 + 激活 + pooling
    with tf.variable_scope("conv2"):
        # 用64个filter 5*5*32
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = biad_variable([64])

        # 从[None,14,14,32] -> [None,14,14,64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # pooling 2*2 ,stride=2, [None,14,14,64] ->[None,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3. FC 全连接层 【None,7,7,64] -> [None,7*7*64]*[7*7*64,10]+[10]=[None,10]
    with tf.variable_scope("FC"):
        # 初始化权重和偏置
        w_fc = weight_variable([7*7*64,10])
        b_fc = biad_variable([10])

        # 对x进行形状改变
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
        y_hat = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_hat


def conv_fc():
    # 获取正式数据：
    mnist = input_data.read_data_sets(
        "/Users/Vincent_Xia/PycharmProjects/leetcode/py_functions/tensorflow_tf/ANN/NN_Mnist", one_hot=True)
    # 定义模型 得到输出
    x, y_true, y_hat = model_DIY()

    with tf.variable_scope("soft_cross"):
        # 求出平均交叉墒损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_hat))

    with tf.variable_scope("optimizer"):
        # 梯度下降求出损失
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # 5 计算正确率
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_hat, 1))
        # equal_list None 个样本[1, 0, 0 , ....,1, 0]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 之前的任务做完了 现在初始化op
    init_op = tf.global_variables_initializer()


    # 开启会话
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init_op)

        # 迭代步数去训练， 更新参数
        for i in range(100):

            # 我先用batch 每次都得到500个训练数据给模型训练 每个batch的数据小的话 数据不一定稳定的 因为样本差别可能很大
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 训练train_op
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
            print(f"训练到第{i}步 之后 准确率 = {sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})}")


if __name__ == "__main__":
    conv_fc()

