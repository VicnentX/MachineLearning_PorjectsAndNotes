import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def csvread(filelist):
    """
    读取csv文件
    filelist：文件路径+名字的列表
    :param filelist:
    :return: 读取的内容
    """

    # 1.构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2.构造csv阅读器读取队列的数据（按一行）
    reader = tf.TextLineReader()    # 阅读器的op叫reader
    key, value = reader.read(file_queue)  # !!!!key is 文件名 ， value is 读取的那一行的内容

    # 3. 对每一行数据进行解码

    # record_defaults: 指定每个样本的每一列的类型， 指定默认值 record_defaults 就是我可以定义这一列按int 还是 string 还是什么 解码
    records = [["None"], ["None"]]

    # decode_csv 返回就是有几列就返回几个值
    example, label = tf.decode_csv(value, record_defaults=records)  # 当然也可以用一个【】 或者 （）接受
    # print(example, label)
    # print(value)


    # 4. 想要读取多个数据 就是批处理

    """
    批处理大小 ， 每批次： 取多少数据和队列和数据没有关系 指由batch——size决定 比capacity大的话 数据就会有重复
    
    一般batch_size is as same as capacity
    
    """
    example_batch, label_batch = tf.train.batch([example, label], batch_size=20, num_threads=1, capacity=9)
    # print(example_batch)
    print(example_batch.shape)
    # print(label_batch)
    print(label_batch.shape)

    return example_batch, label_batch


if __name__ == "__main__":
    path = "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/read_file_API/read_csv_file/csv_data/"
    # 找到文件， 放入列表    路径+名字-》放入列表
    file_name = os.listdir(path)
    # print(file_name)
    filelist = [os.path.join(path, file) for file in file_name]
    print(filelist)
    print(len(filelist))
    example_batch, label_batch = csvread(filelist)

    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
        # 打印读取的内容
        print(sess.run([example_batch, label_batch]))

        # 回收线程
        coord.request_stop()
        coord.join(threads)



