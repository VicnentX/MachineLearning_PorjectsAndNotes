#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])


# 有几个方法让数据都除以每一行的和：

x_edited = x / x.sum(axis=1, keepdims=True)
print("x_edited is like : ", x_edited)
x_edited2 = x / x.sum(axis=1).reshape((x.shape[0], 1))
print("x_edited2 is like : ", x_edited2)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(x, y)

#Predict Output
predicted= model.predict([[1,2],[3,4]])
print("predict result is like : ", predicted)
print("predict result's size : ", predicted.shape)
print("predict result's type : ", type(predicted))