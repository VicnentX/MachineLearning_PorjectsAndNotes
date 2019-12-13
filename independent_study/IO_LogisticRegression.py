import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

raw_df = pd.read_csv('./stock_prices.csv')
# add Date, High, Low, Open, Close, Volume, Adj Close as columns
# these above columns will be used as criteria to classify whether to buy or not
df = raw_df.copy().shift(-1)
# treat today Close price as Buy price
df['Buy'] = raw_df.Close

# High_Delta means  High[i]/High[i - 1]
# Low_Delta means  Low[i]/Low[i - 1]
# Open_Delta means  Open[i]/Open[i - 1]
# Close_Delta means  Close[i]/Close[i - 1]
# Volume_Delta means  Volume[i]/Volume[i - 1]
for item in ['High', 'Low', 'Open', 'Close', 'Volume']:
    df[item + '_Delta'] = raw_df[item] / raw_df[item].shift(1)

# Volume5Sum_Delta means Sum(Volume[i - 4] to Volume[i])/Sum(Volume[i - 9] to Volume[i - 5])
df['Volume5Sum_Delta'] = raw_df.Volume.rolling(5).sum() / (raw_df.Volume.rolling(10).sum() - raw_df.Volume.rolling(5).sum())

# MA means simple moving average
# MA#X# mean simple moving average for X days
# MA5_Delta means MA5[i]/MA5[i - 1]
# MA10_Delta means MA10[i]/MA10[i - 1]
# MA20_Delta means MA20[i]/MA20[i - 1]
# MA40_Delta means MA40[i]/MA40[i - 1]
# MA100_Delta means MA100[i]/MA100[i - 1]
period_list = [5, 10, 20, 40, 100]
for period in period_list:
        df['MA' + str(period) + '_Delta'] = raw_df.Close.rolling(period).mean() / raw_df.Close.shift(1).rolling(period).mean()

# MA#X#cross#Y# means the angle between MA#X# and MA#Y#
# MA5cross10 mean arctan(MA5_Delta - MA10_Delta)/(1 + MA5_Delta * MA10_Delta)
# MA5cross20 mean arctan(MA5_Delta - MA20_Delta)/(1 + MA5_Delta * MA20_Delta)
# MA5cross40 mean arctan(MA5_Delta - MA40_Delta)/(1 + MA5_Delta * MA40_Delta)
# MA5cross100 mean arctan(MA5_Delta - MA100_Delta)/(1 + MA5_Delta * MA100_Delta)
# MA10cross20 mean arctan(MA10_Delta - MA20_Delta)/(1 + MA10_Delta * MA20_Delta)
# MA10cross40 mean arctan(MA10_Delta - MA40_Delta)/(1 + MA10_Delta * MA40_Delta)
# MA10cross100 mean arctan(MA10_Delta - MA100_Delta)/(1 + MA10_Delta * MA100_Delta)
# MA20cross40 mean arctan(MA20_Delta - MA40_Delta)/(1 + MA20_Delta * MA40_Delta)
# MA20cross100 mean arctan(MA20_Delta - MA100_Delta)/(1 + MA20_Delta * MA100_Delta)
# MA40cross100 mean arctan(MA40_Delta - MA100_Delta)/(1 + MA40_Delta * MA100_Delta)
for i in range(len(period_list)):
    for j in range(i + 1, len(period_list)):
        df['MA' + str(period_list[i]) + "cross" + str(period_list[j])] = \
            np.arctan((df['MA' + str(period_list[i]) + '_Delta'] - df['MA' + str(period_list[j]) + '_Delta'])
                      / (1 + df['MA' + str(period_list[i]) + '_Delta'] * df['MA' + str(period_list[j]) + '_Delta']))

# benefit_list to store all the candidate investment combination,
# which looks like [Compound Annual Growth Rate, take profit limit， stop loss limit]
# max_benefit means best Compound Annual Growth Rate
# best_combination means best combination which leads to max_benefit
# peak means the best temp profit in the best_combination

benefit_list = []
max_benefit = 0
best_combination = []
peak = 0
recycle = 0
visualization = pd.DataFrame(columns=['benefit', 'take profit limit', 'stop loss limit'])
best_accuracy = 0
# combination_under_best_accuracy = [best accuracy, benefit, gain, loss]
combination_under_best_accuracy = []

print(df.columns.values)

for gain in np.arange(1.001, 1.05, 0.001):
    for loss in np.arange(0.95, 0.999, 0.001):
        print('this is the ', recycle, ' recycle')
        recycle += 1
        # y = 1 means buy and y = 0 means not buying
        df['y'] = (((df.Open / df.Buy) >= gain) | (((df.High / df.Buy) >= gain) & ((df.Low / df.Buy) >= loss))) * 1
        #remove nan
        xy = df.dropna()
        print('data size is: ', xy.shape)
        # Choose part of X for training
        X = xy.iloc[:, 0:29]
        y = xy.y
        print('counts 1 and 0 in y : ', y.value_counts())
        # split the data into train sets and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # the other way to split the data set in time series
        # X_train = X.iloc[0: 7000]
        # X_test = X.iloc[7000: X.shape[0]]
        # y_train = y.iloc[0: 7000]
        # y_test = y.iloc[7000: y.shape[0]]

        # print('shapes of X_train, X_test, y_train, y_test: ')
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        # instantiate a logistic regression model, and fit with X and y
        logistic_regression_model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=100, verbose=2)
        # only choose the Columns from High_Delta to MA40cross100 to train the model
        logistic_regression_model.fit(X_train.iloc[:, 8:29], y_train)
        # score
        y_test_hat = logistic_regression_model.predict(X_test.iloc[:, 8:29])

        accuracy = logistic_regression_model.score(X_test.iloc[:, 8:29], y_test)
        best_accuracy = max(best_accuracy, accuracy)

        # here, use the Compound Annual Profit to evaluate the mode not the accuracy
        benefit = 1
        temp_best_profit = 0
        for i in range(len(y_test_hat)):
            if y_test_hat[i] == 1:
                # iloc[row, col] , we can not use columns name here for iloc
                # columns - index
                # High - 1
                # Low - 2
                # Open - 3
                # Close - 4
                # Buy - 7
                if (X_test.iloc[i, 3] / X_test.iloc[i, 7]) >= gain:
                    benefit *= gain
                    temp_best_profit = max(benefit, temp_best_profit)
                elif (X_test.iloc[i, 2] / X_test.iloc[i, 7]) <= loss:
                    benefit *= loss
                elif (X_test.iloc[i, 1] / X_test.iloc[i, 7]) >= gain:
                    benefit *= gain
                    temp_best_profit = max(benefit, temp_best_profit)
                else:
                    benefit *= (X_test.iloc[i, 4] / X_test.iloc[i, 7])
                    temp_best_profit = max(benefit, temp_best_profit)

        visualization.loc[visualization.shape[0] + 1] = [benefit, gain, loss]

        if best_accuracy == accuracy:
            combination_under_best_accuracy = [best_accuracy, benefit, gain, loss]

        if benefit > max_benefit:
            max_benefit = benefit
            best_combination = [benefit, gain, loss]
            peak = temp_best_profit

print('shape of visualization = ', visualization.shape)
print('recycle = ', recycle)
print('max_benefit = ', max_benefit)
print('best_combination = ', best_combination)
print('peak = ', peak)
print('combination_under_best_accuracy =', combination_under_best_accuracy)

# plt show()
# ['benefit', 'take profit limit', 'stop loss limit']
threedee = plt.figure().gca(projection='3d')
threedee.scatter(visualization['take profit limit'], visualization['stop loss limit'], visualization['benefit'])
threedee.set_xlabel('take profit limit')
threedee.set_ylabel('stop loss limit')
threedee.set_zlabel('benefit')
plt.show()
