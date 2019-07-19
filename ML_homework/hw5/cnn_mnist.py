# https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Let's start by importing numpy and setting a seed for the computer's pseudorandom number generator.
# This allows us to reproduce the results from our script:
import numpy as np

# Next, we'll import the Sequential model type from Keras.
# This is simply a linear stack of neural network layers,
# and it's perfect for the type of feed-forward CNN we're building in this tutorial.
from keras.models import Sequential

# Next, let's import the "core" layers from Keras.
# These are the layers that are used in almost any neural network:
from keras.layers import Dense, Dropout, Activation, Flatten

# Then, we'll import the CNN layers from Keras.
# These are the convolutional layers that will help us efficiently train on image data:
from keras.layers import Convolution2D, MaxPooling2D

# Finally, we'll import some utilities.
# This will help us transform our data later:
from keras.utils import np_utils

# for drawing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Now we have everything we need to build our neural network architecture.
# load MNIST data
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("now look at the shape of X_train________as below")
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print("now look at the shape of X_train________as below")
print(X_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print("now look at the shape of y_train________as below")
print(y_train.shape)
print(y_train[0])

# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print("now look at the shape of y_train________as below")
print(y_train.shape)
print(y_train[0])

# Let's start by declaring a sequential model format:
model = Sequential()
# CNN input layers
# The input shape parameter should be the shape of 1 sample.
# In this case, it's the same (1, 28, 28) that corresponds to  the (depth, width, height) of each digit image.
# But what do the first 3 parameters represent? They correspond to the number of convolution filters to use,
# the number of rows in each convolution kernel, and the number of columns in each convolution kernel, respectively.
# *Note: The step size is (1,1) by default, and it can be tuned using the 'subsample' parameter.
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding="same"))
print("now look at the shape of model.output_shape________as below")
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print("now look at the shape of model.output_shape________as below")
print(model.output_shape)
model.add(Convolution2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(14,14,32), padding="same"))
print("now look at the shape of model.output_shape________as below")
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print("now look at the shape of model.output_shape________as below")
print(model.output_shape)
model.add(Convolution2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(7,7,64), padding="same"))
print("now look at the shape of model.output_shape________as below")
print(model.output_shape)

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
print("now look at the shape of model.output_shape________as below")
print(model.output_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=10, nb_epoch=5, verbose=1)

y_hat = model.predict(X_test)
print("y_hat shape is as below__________")
print(y_hat.shape)

print(model.metrics_names)
score = model.evaluate(X_test, y_test, verbose=0)
print(f"loss is : {score[0]} and accuracy is : {score[1]}")

# show confuse matrix

# reshape y_hat and y_test
true_label = np.argmax(y_test, axis=1)
predict_label = np.argmax(y_hat, axis=1)

cm = confusion_matrix(true_label, predict_label)
print(cm)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = f"MNIST_with_CNN's Accuracy Score: {score[1]}"
plt.title(all_sample_title, size=15)
plt.show()


