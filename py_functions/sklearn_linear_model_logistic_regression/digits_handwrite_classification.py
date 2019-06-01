# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics

from sklearn.linear_model.sag import sag_solver

digits = load_digits()
print("image data shape: ", digits.data.shape)
print("label data shape: ", digits.target.shape)

# Showing the Images and the Labels (Digits Dataset)
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

# Splitting Data into Training and Test Sets (Digits Dataset)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
print(X_test)

# Scikit-learn 4-Step Modeling Pattern (Digits Dataset)
logistic_regr = LogisticRegression(multi_class="auto", solver="saga", max_iter=100)
logistic_regr.fit(X_train, y_train)

# Predict for One Observation (image)
# 这里很有趣的一点就是要reshape一下 不然我取出来的一行 系统会认为是一个1D array 不是1*M的2D array
print(logistic_regr.predict(X_test[0].reshape(1, -1)))
# Predict for Multiple Observations (images) at Once
print(logistic_regr.predict(X_test[0: 10]))
# Make predictions on entire test data
predictions = logistic_regr.predict(X_test)

# Measuring Model Performance (Digits Dataset)

# Use score method to get accuracy of model
score = logistic_regr.score(X_test, y_test)
print(score)

# Confusion Matrix (Digits Dataset)
# A confusion matrix is a table
# that is often used to describe the performance of a classification model
# (or “classifier”) on a set of test data for which the true values are known.
# In this section, I am just showing two python packages (Seaborn and Matplotlib)
# for making confusion matrices more understandable and visually appealing.

# The confusion matrix below is not visually super informative or visually appealing.

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# Method 1 (Seaborn)
# As you can see below,
# this method produces a more understandable and visually readable confusion matrix using seaborn.

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);
plt.show()