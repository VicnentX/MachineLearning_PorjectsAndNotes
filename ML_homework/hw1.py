import numpy as np
from sklearn.datasets import fetch_california_housing
import xlrd
import pandas as pd
file_name = r"/Users/Vincent_Xia/Downloads/pima-indians-diabetes.xlsx"



# housing = fetch_california_housing()
# m, n = housing.data.shape
#
# print(type(housing))
# print(type(housing))

matrix = np.ones((6, 2))
print(matrix)

dataframe = pd.read_excel(file_name)
print(dataframe)
print(type(dataframe))

matrix = dataframe * dataframe
print(matrix)
print(type(matrix))
