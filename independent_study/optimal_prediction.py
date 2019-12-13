import numpy as np
from datetime import datetime
import pandas_datareader.data as web
from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score


raw_df = pd.read_csv('./stock_prices.csv')
# print(raw_df.shape)
df = raw_df.copy().shift(-1)

df['Buy'] = raw_df.Close

# print(df.shape)
# print(df)

# 完成High_Delta等等， 就是High[i]/High[i - 1]
for item in ['High', 'Low', 'Open', 'Close', 'Volume']:
    df[item + '_Delta'] = raw_df[item] / raw_df[item].shift(1)

# print(raw_df.shape)
# print(df.shape)
# print(df)

# print(df)
df['Volume5Sum_Delta'] = raw_df.Volume.rolling(5).sum() / (raw_df.Volume.rolling(10).sum() - raw_df.Volume.rolling(5).sum())
# print(df.shape)
# print(df)

period_list = [5, 10, 20, 40, 100]
for period in period_list:
        df['MA'+ str(period) +'_Delta'] = raw_df.Close.rolling(period).mean() / raw_df.Close.shift(1).rolling(period).mean()

# print(df.shape)
# print(df.MA5_Delta)

for i in range(len(period_list)):
    for j in range(i + 1, len(period_list)):
        df['MA'+ str(period_list[i]) + "cross" + str(period_list[j])] = \
            np.arctan((df['MA' + str(period_list[i]) + '_Delta'] - df['MA' + str(period_list[j]) +'_Delta']) / (1 + df['MA' + str(period_list[i]) + '_Delta'] * df['MA' + str(period_list[j]) + '_Delta']))

# print(df.shape)
# print(df)

benefit_list = []
max_benefit = 0
best_combination = []
recycle = 0

for gain in np.arange(1.001, 1.05, 0.001):
    for loss in np.arange(0.95, 0.999, 0.001):
        # print(str(gain) + " " + str(loss))
        print('这里是第 ', recycle, ' 次循环')
        recycle += 1
        df['y'] = (((df.Open / df.Buy) >= gain) | (((df.High / df.Close) >= gain) & ((df.Low / df.Close) >= loss))) * 1
        # print(df.y.value_counts())
        #去掉nan
        xy = df.dropna()
        # print("XXXXXYYYYYYY")
        # print(xy.shape)
        # print(xy)
        # print(xy.shape)
        # 只选取需要的做training
        X = xy.iloc[:, 0:29]
        # print('X的下面这个样子的', X.shape,  X)

        y = xy.y
        # print('y的下面这个样子的', y.shape, y)
        # print('y里面1和0的个数', y.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # # 还有一种做法就是随机切割，就是选取上面的连续的作为train 留下来的作为test
        # X_train = X.iloc[0: 7000]
        # X_test = X.iloc[7000: X.shape[0]]
        # y_train = y.iloc[0: 7000]
        # y_test = y.iloc[7000: y.shape[0]]


        # print('X_train, X_test, y_train, y_test 这四个的形状是')
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        # instantiate a logistic regression model, and fit with X and y
        logistic_regression_model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=100, verbose=2)
        # logistic_regression_model = LogisticRegression()
        # print('X_train.iloc[:, 8:28]的下面这个样子的', X_train.iloc[:, 8:29].shape, X_train.iloc[:, 8:29])
        logistic_regression_model.fit(X_train.iloc[:, 8:29], y_train)
        # score
        y_test_hat = logistic_regression_model.predict(X_test.iloc[:, 8:29])
        # print("y_test_hat 的 类型", type(y_test_hat))
        # print(y_test.shape)
        # accuracy = logistic_regression_model.score(X_test, y_test)
        # print("accuracy ： ", accuracy)


        # print("X_test is looks like: ", X_test.shape)
        # print(X_test)
        # print("y_test is looks like: ", y_test.shape)
        # print(y_test)

        benefit = 1
        for i in range(len(y_test_hat)):
            # print('I am in the for loop to calcualte the benefit rate')
            if y_test_hat[i] == 1:
                # print("________________")
                # print(X_test.columns)
                # print("________________")
                # 这里好像只能用iloc加上index  注意 iloc是要【行，列】 。。。。。。。用列名会报错 。。。。。。
                # 列名字 - 数字
                # High - 1
                # Low - 2
                # Open - 3
                # Close - 4
                # Buy - 7

                if (X_test.iloc[i, 3] / X_test.iloc[i, 7]) >= gain:
                    benefit *= gain
                elif (X_test.iloc[i, 2] / X_test.iloc[i, 7]) <= loss:
                    benefit *= loss
                elif (X_test.iloc[i, 1] / X_test.iloc[i, 7]) >= gain:
                    benefit *= gain
                else:
                    benefit *= (X_test.iloc[i, 4] / X_test.iloc[i, 7])

        benefit_list.append([benefit, gain, loss])

        if benefit > max_benefit:
            max_benefit = benefit
            best_combination = [benefit, gain, loss]

print('recycle = ', recycle)
print('max_benefit is ', max_benefit)
print('best_combination is ', best_combination)
print(len(benefit_list))




