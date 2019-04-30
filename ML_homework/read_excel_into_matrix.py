import xlrd

path = r"/Users/Vincent_Xia/Downloads/pima-indians-diabetes.xlsx"

data = xlrd.open_workbook(path)
table = data.sheets()[0]
m = table.nrows
n = table.ncols
print(m, n)