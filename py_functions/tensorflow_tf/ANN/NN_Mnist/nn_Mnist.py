from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



mnist = input_data.read_data_sets("/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/py_functions/tensorflow_tf/ANN/NN_Mnist", one_hot=True)
print(mnist.train.images)
print(mnist.train.images.shape)
print(mnist.train.labels)
print(mnist.train.labels.shape)
print(mnist.train.next_batch(50))
print(type(mnist.train.next_batch(50)))