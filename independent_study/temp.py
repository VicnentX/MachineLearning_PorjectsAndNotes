import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.DataFrame({'code':[1,2,3,4,5,6,7,8],
                   'value':[np.nan,5,7,8,9,10,11,12],
                   'value2':[5,np.nan,7,np.nan,9,10,11,12],
                   'indstry':['农业1','农业1','农业1','农业2','农业2','农业4','农业2','农业3']},
                    columns=['code','value','value2','indstry'])

mean = df['code'][max(0, 1):3].mean()
sum = df['code'][max(0, 1):3].sum()

print('--------------------')

print(df.code[1])

print('--------------------')
print(mean)
print(sum)
print('--------------------')
df['new'] = 0
print(df)
df['new'][0] = 1
print(df)
print(df.shape[0])

df['new'] = [(df['code'][max(0, i - 2): i + 1].mean() / df['code'][max(0, i - 3): i].mean()) for i in range(df.shape[0])]
print(df)

m1 = -1000000000000000
m2 = 0

angle_tan = (m1 - m2) / (1 + m1 * m2)
angle = np.arctan(angle_tan)
print(angle)

for i in range(1, 11, 1):
    for j in range(-10, 0, 1):
        print(i, " ", j)

# loc 里面的冒号是包含的
df2 = df.loc[0:df.shape[0],['code','value']]
print(df2)
print(type(df2))

df3 = df.loc[0:df.shape[0],'code':'indstry']
print(df3)
print(df3.shape)

# iloc 里面的冒号是不包含的
df4 = df.iloc[1:4,[0,2]]
print(df4)

df.loc[5:7,'newnew'] = df.loc[5:7,'code'] + df.loc[5:7,'value']
print(df)


list = []
list.append([1,2,3])
list.append([4,5,6])
list.append([7,8,9])
df = pd.DataFrame(columns=['benefit', 'take profit limit', 'stop loss limit'])
df.loc[df.shape[0] + 1] = [1,2,3]
df.loc[df.shape[0] + 1] = [4,5,6]
df.loc[df.shape[0] + 1] = [7,8,9]

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df['take profit limit'], df['stop loss limit'], df['benefit'])
threedee.set_xlabel('take profit limit')
threedee.set_ylabel('stop loss limit')
threedee.set_zlabel('benefit')
plt.show()
