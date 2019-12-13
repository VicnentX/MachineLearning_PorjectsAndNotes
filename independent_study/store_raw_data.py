from datetime import datetime
import pandas_datareader.data as web

# get the data
sp500_1989_2019 = web.DataReader("^GSPC", "yahoo", datetime(1989, 1, 1), datetime(2019, 12, 6))
# store the data
sp500_1989_2019.to_csv(r'/Users/Vincent_Xia/PycharmProjects/MachineLearningProjects&Notes/independent_study/stock_prices.csv')
# Don't forget to add '.csv' at the end of the path
