"""
如何确定axis的值，只需要记住axis赋值的维度是要被压缩的维度，如果要得到各列的最大值，需要压缩行这个维度。
"""

from sklearn.preprocessing import normalize
import numpy as np

data = np.array([
    [1000, 10, 0.5],
    [765, 5, 0.35],
    [800, 7, 0.09], ])
data = normalize(data, axis=0, norm="max")
print(data)
