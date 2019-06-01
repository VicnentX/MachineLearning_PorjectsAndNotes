# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:55:52 2019

@author: yingl
"""

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, Y, \
                                                    stratify=Y, random_state=42,test_size=0.25)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8,  activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(1,  activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X_test)
# round predictions
y_test_hat = [round(x[0]) for x in predictions]

num_correct = 0
for i in range(len(y_test)):
    if y_test_hat[i]==y_test[i]:
        num_correct += 1
        
Accuracy_rate = num_correct/len(y_test)
print("Accuracy Rate = ", Accuracy_rate)

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'b-', linewidth=4)
plt.ylabel('True Positive Rate', fontsize=26)
plt.xlabel('False Positive Rate', fontsize=26)

plt.tick_params(labelsize=26)
plt.grid(True)
plt.show()