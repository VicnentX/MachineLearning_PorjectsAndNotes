# https://www.cs.toronto.edu/~kriz/cifar.html

# https://blog.csdn.net/qq_33039859/article/details/79903547

import tensorflow as tf
import os


# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_dir", "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/read_file_API/read_bytes_file/cifar-10-batches-bin/", "文件的目录")



class CifarRead(object):
    """
    完成二进制文件 ， 写进tfrecords 读取tfrecords
    """
    def __init__(self, filelist):
        self.file_list = filelist
        self.height = 32
        self.width = 32
        self.channel = 3
        # 二进制文件一张存储的字节
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):
        """
            注意！！！！！
            1-6这套东西是给子线程做的 主线程是拿这些数据来训练的
        """
        # 1. 构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2. 构造文件读取器去读取二进制文件内容（1 + 3072 = 3073 个 bytes）
        reader = tf.FixedLengthRecordReader(self.bytes)
        key, value = reader.read(file_queue)
        print("——————————————————下面就是value 就是一行数据一个label + 这张图片所有想素点 所以形状shape=（）———————————")
        print(value)
        # 3. 二进制文件解码内容
        label_and_image = tf.decode_raw(value, tf.uint8)
        print("——————————————————下面就是label_and_image 就是一行数据一个label + 这张图片所有想素点 所以形状shape=（）———————————")
        print(label_and_image)

        # 4. 分割label 和 image
        # tf。clice的方法
        # https://blog.csdn.net/qq_33039859/article/details/79903547

        label = tf.cast(tf.slice(label_and_image, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_and_image, [self.label_bytes], [self.image_bytes])
        print("——————————————————下面就是label 和 image -------------")
        print(label, image)

        # 5. reshape the image(32 , 32, 3) 这边注意因为image形状里面没有问号？ 所以只能reshape
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        print("——————————————————下面就是label 和 image_reshape -------------")
        print(label, image_reshape)


        # 6. batch processing
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)
        print("——————————————————下面就是image_batch 和 label_batch -------------")
        print(image_batch, label_batch)

        return image_batch, label_batch


if __name__ == "__main__":

    # 找到文件， 放入列表    路径+名字-》放入列表
    file_name = os.listdir(FLAGS.cifar_dir)
    # print(file_name)
    """
    这里我指读取的5 个 train文件 ，这个文件名的特征是。bin结尾
    """
    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    cf = CifarRead(filelist)
    image_batch, label_batch = cf.read_and_decode()

    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
        # 打印读取的内容
        print("-----正在读取一次-----")
        print(sess.run([image_batch, label_batch]))

        # 回收线程
        coord.request_stop()
        coord.join(threads)
