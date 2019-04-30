import imageio
import Image
import os
from PIL import Image
import numpy as np


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
    if cnt <= 1: continue   # skip .DS.store file

    for img in os.listdir(path + r"/" + lis):

        pic_square = Image.open(path + r"/" + lis + r"/" + img)
        pic_square = np.array(pic_square)
        pic_line = np.reshape(pic_square, (1, pic_square.size))

        if img == "1.pgm" or img == "3.pgm" \
                or img == "4.pgm" or img == "5.pgm" or img == "7.pgm" or img == "9.pgm":

            print("-------------------info about img")
            print(img)
            print(type(img))
            print("-------------------info about pic_square")
            # pic_square = Image.open(path + r"/" + list + r"/" + img)
            print(type(pic_square))
            # pic_square = np.array(pic_square)
            print(type(pic_square))
            print(pic_square.shape)
            print(pic_square.size)
            print("-------------------info about pic_line")
            # pic_line = np.reshape(pic_square, (1, pic_square.size))
            print(type(pic_line))
            print(pic_line.shape)
            print(pic_line.size)

            x_train[index_train] = pic_line
            index_train += 1
        else:
            x_test[index_test] = pic_line
            index_test += 1

    print(x_train.shape)    # should be 61 * 10304
    print(x_test.shape)     # should be 41 * 10304

    print("x_train is :")
    print(x_train)
    print("x_test is : ")
    print(x_test)




