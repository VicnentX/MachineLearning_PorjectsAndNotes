import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def picread(filelist):
    """
    读取狗的图片并转换成张亮
    :param filelist: 文件路径+名字列表
    :return: 每张图片的张亮
    """

    """
    注意！！！！！
    1-6这套东西是给子线程做的 主线程是拿这些数据来训练的
    """
    # 1. 构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2. 构造文件读取器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    print("——————————————————下面是解码之前的value———————————")
    print(value)

    # 3. 对图片数据进行解码
    image = tf.image.decode_jpeg(value) # 注意value是一张图片的value image也是指一张图片
    print("——————————————————下面是解码后image信息———————————")
    print(image)    # shape = (?, ?, ?) 因为长，宽， 通道 都没有固定

    # 4. 处理图片大小 统一大小
    image_resize = tf.image.resize_images(image, [150, 150])   # 这里data type从int8 变成了 float ，所以变回图片的时候要变成int8
    print("——————————————————下面是解码后 加上 resize 之后 image信息 通道依旧是？ 没有确定———————————")
    print(image_resize)

    # 5. 要把第三维也确定下来，因为后面我要把所有的图片都读取出来，
    # 因为如果不知道每个图片的通道数，那么计算机就不知道是一层合成一张（黑白） 还是三层合成一张（彩色）
    image_resize.set_shape([150, 150, 3])
    print("——————————————————下面是解码后 加上 resize 之后 加上 set_shape之后的 image信息 ———————————")
    print(image_resize)

    # 6. 进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=5, num_threads=1, capacity=5)
    print("——————————————————下面是 image_batch 信息 ———————————")
    print("image_batch 的 形状 ： ", image_batch.shape)
    print(image_batch)

    return image_batch


if __name__ == "__main__":
    path = "/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/read_file_API/read_img_file/dog_pics/"
    # 找到文件， 放入列表    路径+名字-》放入列表
    file_name = os.listdir(path)
    # print(file_name)
    filelist = [os.path.join(path, file) for file in file_name]
    print(filelist)
    print(len(filelist))
    image_batch = picread(filelist)

    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读取文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
        # 打印读取的内容
        print("-----正在读取一次-----")
        print("image_resize 's TYPE is :", image_batch)
        print("image_resize 's SHAPE is :", image_batch.shape)
        print(sess.run([image_batch]))

        # 回收线程
        coord.request_stop()
        coord.join(threads)