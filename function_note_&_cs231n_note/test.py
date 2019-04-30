import math
import numpy as np


def dist(x, y):
    return np.linalg.norm(a - b)


num = math.fabs(-6)
print(num)
print(type(num))

a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
distance = dist(a, b)
print(distance)
print(distance ** 2)
print(math.sqrt(1 + 4 + 9))