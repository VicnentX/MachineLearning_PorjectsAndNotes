import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


# 有内置函数可以直接算出来cluster 但是这不是我们缩需要的 不过我还是列出来在这里
from sklearn.cluster import KMeans



file_name = r"Iris.xls"

# read excel file
df = pd.read_excel(file_name)
# print("df's type is ", type(df))
# print(len(df.columns))
raw_data = np.array(df)
raw_data = raw_data[:, 1:raw_data.shape[1] - 1]

# row and col count
row_cnt, col_cnt = raw_data.shape
print("row and col count: ", row_cnt, col_cnt)



# 有内置函数可以直接算出来cluster
# 但是这不是我们缩需要的 不过我还是列出来在这里

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(raw_data[:, 1:col_cnt])
# Getting the cluster labels
labels = kmeans.predict(raw_data[:, 1:col_cnt])
# Centroid values
centroids = kmeans.cluster_centers_

print(centroids)
#
# # 以上就是内置函数给我们算出来的clusters



# print(raw_data)

x_plot = []
y_plot = []


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# Number of clusters
k = 3
#
index_of_orignal_cluster = np.random.randint(0, 150, 3)
print(index_of_orignal_cluster)
c = np.array(raw_data[index_of_orignal_cluster])
print(c)

# cluster labels(0, 1, 2)
clusters = np.zeros(row_cnt)

# to assign 0 to the J_old, 100 to error(|J_cur - J_old|)
j_cur = 0
error = 1
run = 0

while run < 1012 and error > math.pow(10, -5):
    # assigning each value to its closest cluster
    j_old = j_cur
    j_cur = 0
    for i in range(row_cnt):
        distances = dist(raw_data[i], c)
        print(type(distances), distances)
        cluster_index_min = np.argmin(distances)  # int
        j_cur += distances[cluster_index_min] ** 2
        clusters[i] = cluster_index_min   # float clusters store the index for i row data
        print(cluster_index_min, clusters[i])
        print("____________")
    # storing the old centroid values  这里没有用到以前的点
    c_old = c.copy()
    # finding the new centorids by taking the average value
    for i in range(k):
        points = [raw_data[j] for j in range(row_cnt) if clusters[j] == i]
        c[i] = np.mean(points, axis=0)

    error = math.fabs(j_cur - j_old)
    run += 1
    x_plot.append(run)
    y_plot.append(j_cur)

print("x_plot is : ", x_plot)
print("y_plot is : ", y_plot)

plt.plot(x_plot, y_plot, "*--")
plt.xlabel("n: iteration count")
plt.ylabel("J: objective function")
plt.title("J vs n")
plt.legend(["objective function value J curve"])
plt.show()
