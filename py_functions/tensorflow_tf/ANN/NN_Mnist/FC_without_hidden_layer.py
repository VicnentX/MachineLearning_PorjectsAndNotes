import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("is_train", 1, "指定程序是去做预测 还是 训练")


def full_connected():

    # 获取正式数据：
    mnist = input_data.read_data_sets("/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/ANN/NN_Mnist", one_hot=True)

    # 1。 建立数据占位符 x [None, 784] y_true [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2. 建立全链接层的神经网络 W[784, 10] B[10]
    with tf.variable_scope("fc_model"):
        # 随机初始化权重和偏执
        W = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0, name="W"))
        B = tf.Variable(tf.constant(0.0, shape=[10]))

        # 预测输出None个样本输出结果 x * W + B = [None, 10]
        y_hat = tf.matmul(x, W) + B

    with tf.variable_scope("soft_cross"):
        # 求出平均交叉墒损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_hat))

    with tf.variable_scope("optimizer"):
        # 梯度下降求出损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5 计算正确率
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_hat, 1))
        # equal_list None 个样本[1, 0, 0 , ....,1, 0]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 之前的任务做完了 现在初始化op
    init_op = tf.global_variables_initializer()

    """
    这里要创建一个ssaver去保存训练完之后的model
    """
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init_op)


        if FLAGS.is_train == 1:
            # 迭代步数去训练， 更新参数
            for i in range(2048):

                # 我先用batch 每次都得到500个训练数据给模型训练 每个batch的数据小的话 数据不一定稳定的 因为样本差别可能很大
                mnist_x, mnist_y = mnist.train.next_batch(500)

                # 训练train_op
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                print(f"训练到第{i}步 之后 准确率 = {sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})}")

            """
            保存之前训练完之后的模型，model的名字需要写
            """
            saver.save(sess, "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/ANN/NN_Mnist/skpt/fc_model")
        else:
            """
            加载之前save的模型
            """
            saver.restore(sess, "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/ANN/NN_Mnist/skpt/fc_model")
            for i in range(100):
                # 每次测试一张图片
                x_test, y_test = mnist.test.next_batch(1)
                print(f"第{i}张图片，手写的数字目标是{tf.argmax(y_test, 1).eval()}, 我的预测是{tf.argmax(sess.run(y_hat, feed_dict={x: x_test, y_true: y_test}), 1).eval()}")



    return None


if __name__ == "__main__":
    full_connected()