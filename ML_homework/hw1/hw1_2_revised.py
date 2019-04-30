import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_name = r"pima-indians-diabetes.xlsx"

# read excel file
df = pd.read_excel(file_name)
# print("df's type is ", type(df))
# print(len(df.columns))
raw_data = np.array(df)
print("matrix's type is ", type(raw_data))
# print(raw_data)
row_cnt, col_cnt = raw_data.shape
raw_data_plus_bias = np.c_[np.ones(row_cnt), raw_data]

mul = np.array([1,1,1,1,1,1,1,1,1,10])
raw_data_plus_bias = raw_data_plus_bias * mul

row_cnt, col_cnt = raw_data_plus_bias.shape
print("row_cnt and col_cnt is : ", row_cnt, col_cnt)


x_plot = []
y_plot = []

for n in range(20, 101, 20):
    ave_accuracy_rate = 0

    for iter in range(1000):
        test_data_1 = set()
        test_data_0 = set()
        t = np.zeros((2 * n, 1))
        x = np.ones((2 * n, col_cnt - 1))  # 这里－1是因为有outcome那列不使用
        while len(test_data_0) < n or len(test_data_0) < n:
            index = np.random.randint(row_cnt)
            # print(type(index))
            # print(index)
            point = len(test_data_0) + len(test_data_1)
            # print("len(test_data_0) is ", len(test_data_0))
            # print("len(test_data_1) is ", len(test_data_1))
            # print("point is ", point)
            if raw_data_plus_bias[index, col_cnt - 1] == 0 and len(test_data_0) < n and index not in test_data_0 :
                test_data_0.add(index)
                x[point, :] = raw_data_plus_bias[index, :col_cnt - 1]
                t[point] = 0
            if raw_data_plus_bias[index, col_cnt - 1] == 1 and len(test_data_1) < n and index not in test_data_1:
                test_data_1.add(index)
                x[point, :] = raw_data_plus_bias[index, :col_cnt - 1]
                t[point] = 1
        # have succeed to get x, t right now
        xt = x.T
        w = np.linalg.inv((xt.dot(x))).dot(xt).dot(t)
        wt = w.T
        # print("wt's shape is ", wt.shape)

        correct_prediction_case_cnt = 0
        for i in range(row_cnt):
            if i in test_data_1 or i in test_data_0:
                continue
            # y_predicted = wt.dot(np.insert(raw_data[i, 0:col_cnt - 1], values=np.ones(1,), axis=1))
            y_predicted = wt.dot(raw_data_plus_bias[i, 0:col_cnt - 1])
            y = 10 if y_predicted >= 5 else 0
            if y == raw_data_plus_bias[i, col_cnt - 1]:
                correct_prediction_case_cnt += 1

        # calculate accuracy rate
        ave_accuracy_rate += correct_prediction_case_cnt / (row_cnt - 2 * n)

    ave_accuracy_rate /= 1000
    x_plot.append(n)
    y_plot.append(ave_accuracy_rate)

print("x_plot is : ", x_plot)
print("y_plot is : ", y_plot)

plt.plot(x_plot, y_plot, "*--")
plt.xlabel("n: test case for each diabetes and non-diabetes")
plt.ylabel("accuracy prediction rate")
plt.title("accuracy vs n")
plt.legend(["accuracy rate curve"])
plt.show()






