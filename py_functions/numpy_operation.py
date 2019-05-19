import numpy as np

data = np.array([[1,1,1],[2,2,2],[3,3,3]])
vector = np.array([1,2,3])
print("vector shape is like : ", vector.shape)

sub_result = data - vector[:, None]
print("sub_result is like :", sub_result)
div_result = data / vector[:, None]
print("div_result is like :", div_result)

vector_t = vector.reshape((1,3))
print("vector shape is like : ", vector_t.shape)
sub_result = data - vector_t[:, None]
print("sub_result is like :", sub_result)
div_result = data / vector_t[:, None]
print("div_result is like :", div_result)

"""
每一行除以这一行的sum
"""
new_result = data / data.sum(axis=1).reshape(3,1)
print("每一行除以这一行的sum 所得到的最后矩阵")
print(new_result)


a = np.diag([10,20,30,40])
print("a:")
print(a)
b = np.diag([10,20,30,40], 2)
print("b:")
print(b)
print("b size")
print(b.size)
print("b shape")
print(b.shape)
print("b dimens")
print(b.ndim)
print("b type")
print(b.dtype)
print("length of each axis")
for i in range(b.ndim):
    print(b.shape[i])

array_random = np.random.rand(4,4)
print(array_random)

"""
conditional operation
"""
score = np.array([[80, 88], [82, 81], [84, 75], [86, 83], [75, 81]])
# 如果数值小于80，替换为0，如果大于等于80，替换为90
re_score = np.where(score < 80, 0, 90)
print(re_score)
"""
统计运算
"""
# 求整个矩阵的最大值
result = np.amax(score)
print(result)
# 求每一列的最大值（0表示行）
result = np.amax(score, axis=0)
print(result)
# 求每一行的最大值（1表示列）
result = np.amax(score, axis=1)
print(result)
print(result.shape)

# 求整个矩阵的最小值
result = np.amin(score)
print(result)
# 求每一列的最小值（0表示行）
result = np.amin(score, axis=0)
print(result)
# 求每一行的最小值（1表示列）
result = np.amin(score, axis=1)
print(result)

# 求整个矩阵的平均值
result = np.mean(score, dtype=np.int)
print(result)
# 求每一列的平均值（0表示行）
result = np.mean(score, axis=0)
print(result)
# 求每一行的平均值（1表示列）
result = np.mean(score, axis=1)
print(result)

# 求整个矩阵的方差
result = np.std(score)
print(result)
# 求每一列的方差（0表示列）
result = np.std(score, axis=0)
print(result)
# 求每一行的方差（1表示行）
result = np.std(score, axis=1)
print(result)
print(result.shape)
tem = result.reshape(1, 5)
print(tem.shape)

a1 = np.array([[1, 2], [2, 3]])
a2 = np.array([[2, 3], [3, 4]])
print(np.intersect1d(a1, a2))
print(np.setdiff1d(a1, a2))
print(np.union1d(a1, a2))

st_score = np.array([[80, 88], [82, 81], [84, 75], [86, 83], [75, 81]])
# 平时成绩占40% 期末成绩占60%, 计算结果
q = np.array([[0.4], [0.6]])
result = np.dot(st_score, q)
print(result)
print(result.shape)


# matrix 拼接
v1 = [[0, 1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10, 11]]
v2 = [[12, 13, 14, 15, 16, 17],
      [18, 19, 20, 21, 22, 23],
      [18, 19, 20, 21, 22, 23]]
result = np.vstack((v1, v2))
print(result)

v1 = [[0, 1, 2, 3, 4, 5],
       [6, 7, 8, 9, 10, 11]]
v2 = [[12, 13, 14, 15, 16, 17],
      [18, 19, 20, 21, 22, 23]]
result = np.hstack((v1, v2))
print(result)

"""
numpy delete
"""
OriginalY = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
print(np.delete(OriginalY, [0, 2]))
print(np.delete(OriginalY, [0, 2], axis=0))
print(np.delete(OriginalY, [0, 2], axis=1))

"""
append
"""
OriginalY = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
# 末尾添加元素
print(np.append(OriginalY, [0, 2]))
# 最后一行添加一行
print(np.append(OriginalY, [[0, 2, 11]], axis=0))
# 最后一列添加一列（注意添加元素格式）
print(np.append(OriginalY, [[0], [2], [11]], axis=1))

"""
矩阵插入：Numpy.insert
(参数 1：array，数组；参数 2：index，插入位置索引；
参数 3： elements，添加元素；参数 4： axis=0/1)
"""
OriginalY = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
print(np.insert(OriginalY, 1, [11, 12, 10]))
print(np.insert(OriginalY, 1, [[11, 12, 10]], axis=0))
# 在列索引1的位置插入（注意元素格式，跟添加格式不同）
print(np.insert(OriginalY, 1, [[11, 12, 10]], axis=1))