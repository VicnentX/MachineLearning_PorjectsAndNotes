"""
https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
"""


import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from matplotlib import image as img
from sklearn import metrics
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# read train image
train_img = Image.open("family.jpg")
print(type(train_img))
train_img = np.array(train_img)
print("train_img.shape : ", train_img.shape)
print("train_image is like : ")
print(train_img)
x_train = np.reshape(train_img, (-1, 3))
print("----------------------")
print(x_train)
print("----------------------")
print("x_train.shape : ", x_train.shape)

# read train ground truth
train_ground_img = Image.open("family.png")
train_ground_img = np.array(train_ground_img)
y_train = np.reshape(train_ground_img, (512 * 640, 4))

# read test image
test_img = Image.open("portrait.jpg")
test_img = np.array(test_img)
print(test_img.shape)
x_test = np.reshape(test_img, (-1, 3))
print(x_test.shape)

# read test ground truth
test_ground_img = Image.open("portrait.png")
test_ground_img = np.array(test_ground_img)

img.imsave("portrait_convert.png", test_ground_img)

print(test_ground_img.shape)
y_test = np.reshape(test_ground_img, (-1, 4))
print(y_test.shape)

# 0 means black or bg , 1 means white or skin
print(set(y_train[i][0] for i in range(y_train.shape[0])))
print(set(y_test[i][0] for i in range(y_test.shape[0])))

"""
here I ues two method to do classification
method1 - use RGB (3 independent features)
method2 - ues normalize then only use r=R/R + G + B, g=G/R + G + B
"""


#                        method 1
print("---method1 - use RGB (3 independent features)---")
model1 = GaussianNB()
model1.fit(x_train, y_train[:, 0])
predict1 = model1.predict(x_test)

# save the photo
predict1 = predict1.reshape(predict1.size, 1)
predict1 = np.repeat(predict1, 3, axis=1)
# 这个照片只用rgb 没有第四维
img.imsave("predict_portrait_3_ndim.png",
           predict1.reshape(test_ground_img.shape[0],test_ground_img.shape[1],3))

# 这个照片用rgb+饱和度 有第四维
predict1 = np.append(predict1, np.full((predict1.shape[0], 1), 255), axis=1)
img.imsave("predict_portrait_RGB.png", predict1.reshape(test_ground_img.shape))

# data analysis builtin
print(metrics.classification_report(y_test[:, 0], predict1[:, 0]))

# print confusion matrix
cm1 = metrics.confusion_matrix(y_test[:, 0], predict1[:, 0])
print(cm1.dtype)

cm1_normalized = cm1.astype("float") / cm1.sum(axis=1).reshape(-1, 1)
print(cm1_normalized)
print("true_positive_rate = ", cm1_normalized[1, 1])
print("true_negative_rate = ", cm1_normalized[0, 0])
print("false_positive_rate = ", cm1_normalized[0, 1])
print("false_negative_rate = ", cm1_normalized[1, 0])
plot_confusion_matrix(cm1, ["background", "skin"], title="Normalized Confusion Matrix with RGB")
plt.show()


#                               method 2
print("---method2 - ues normalize then only use r=R/R + G + B, g=G/R + G + B---")
model2 = GaussianNB()
# add 1 to the element to avoid divide by zero
x_train = x_train + 1

model2.fit((x_train / x_train.sum(axis=1, keepdims=True))[:, 0:2], y_train[:, 0])
predict2 = model2.predict((x_test / x_test.sum(axis=1, keepdims=True))[:, 0:2])
print(predict2.shape)
# save the photo
predict2 = predict2.reshape(-1, 1)
predict2 = np.repeat(predict2, 3, axis=1)
predict2 = np.append(predict2, np.full((predict2.shape[0], 1), 255), axis=1)
img.imsave("predict_portrait_RG_proportion.png", predict2.reshape(test_ground_img.shape))

# print confusion matrix
cm2 = metrics.confusion_matrix(y_test[:, 0], predict2[:, 0])

cm2_normalized = cm2 / cm2.sum(axis=1).reshape(-1, 1)
print(cm2_normalized)
print("true_positive_rate = ", cm2_normalized[1, 1])
print("true_negative_rate = ", cm2_normalized[0, 0])
print("false_positive_rate = ", cm2_normalized[0, 1])
print("false_negative_rate = ", cm2_normalized[1, 0])
plot_confusion_matrix(cm2, ["background", "skin"], title="Normalized Confusion Matrix with RG proportion")
plt.show()

