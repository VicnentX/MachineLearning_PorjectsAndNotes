import imageio
import Image
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# img = Image.open("/Users/Vincent_Xia/PycharmProjects/leetcode/ML_homework/hw2/att_faces_10/s1/1.pgm")
# img.show()
# print(img.size)
# print(type(img))

path = r"att_faces_10"

dirs = os.listdir(path)
cnt = 0

x_train = np.zeros((60, 10304))
x_test = np.zeros((40, 10304))
index_train = 0
index_test = 0

for lis in dirs:
    cnt += 1
    print(lis)
    if cnt == 1:
        continue  # skip .DS.store file

    for img in os.listdir(path + r"/" + lis):

        pic_square = Image.open(path + r"/" + lis + r"/" + img)
        pic_square = np.array(pic_square)
        pic_line = np.reshape(pic_square, (1, pic_square.size))

        if img == "1.pgm" or img == "3.pgm" \
                or img == "4.pgm" or img == "5.pgm" or img == "7.pgm" or img == "9.pgm":

            # print("-------------------info about img")
            # print(img)
            # print(type(img))
            # print("-------------------info about pic_square")
            # # pic_square = Image.open(path + r"/" + list + r"/" + img)
            # print(type(pic_square))
            # # pic_square = np.array(pic_square)
            # print(type(pic_square))
            # print(pic_square.shape)
            # print(pic_square.size)
            # print("-------------------info about pic_line")
            # # pic_line = np.reshape(pic_square, (1, pic_square.size))
            # print(type(pic_line))
            # print(pic_line.shape)
            # print(pic_line.size)

            x_train[index_train] = pic_line
            index_train += 1
        else:
            x_test[index_test] = pic_line
            index_test += 1

print("x_train shape", x_train.shape)   # should be 60 * 10304
print("x_test shape", x_test.shape)     # should be 40 * 10304

    # print("x_train is :")
    # print(x_train)
    # print("x_test is : ")
    # print(x_test)


# print("inbuilt PCA -----------------------------")
#
# print("---------class pca ")
# pca = PCA(n_components=4)
# print(pca.fit(x_train))
#
# print("-----------------------lamdas(特征值的集合) is as : ")
# wk = pca.components_
# print("lamdas is : ")
# print(wk)
# print("lamdas's shape : ")
# print(wk.shape)
#
# print("-----------------------xk(x_reduction) is as : ")
# x_reduction = pca.transform(x_train)    # 相当于xk = x_reduciton
# # print(x_reduction)
# print(x_reduction.shape)

x_plot = []
y_plot = []

"""
k could not be larger than the count of test sample
"""
for k in [1, 2, 3, 6, 10, 20, 30, 50]:
    print(k)

    # generate the x_train_reduction and x_test_reduction
    pca = PCA(n_components=k)
    pca.fit(x_train)
    # wk = pca.components_
    x_train_reduction = pca.transform(x_train)
    print("--------x_train_reduction is below-------")
    print("x_train_reduction shape", x_train_reduction.shape)

    pca.fit(x_train)
    x_test_reduction = pca.transform(x_test)
    print("--------x_test_reduction is below-------")
    print("x_test_reduction shape", x_test_reduction.shape)

    total_case = 0  # total case in each loop of k, it should be 40
    hit_case = 0    # cnt of hit case

    # calculate the field of classification
    for i in range(x_test_reduction.shape[0]):
        distances = dist(x_test_reduction[i], x_train_reduction)
        face_PCA_index = np.argmin(distances)
        if face_PCA_index // 6 == i // 4:
            hit_case += 1
        total_case += 1

    # store pairs of field against k
    x_plot.append(k)
    y_plot.append(hit_case / total_case)

print("x_plot is : ", x_plot)
print("y_plot is : ", y_plot)

plt.plot(x_plot, y_plot, "*--")
plt.xlabel("k: reduced dimens")
plt.ylabel("yield: accuracy rate")
plt.title("yield vs k")
plt.legend(["accuracy rate curve"])
plt.show()
