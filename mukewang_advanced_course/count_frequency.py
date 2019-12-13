from random import randint

data = [randint(0, 20) for _ in range(30)]
print(data)

dict1 = dict.fromkeys(data, 0)
print(dict1)

# 计算frequency 遇到自加1
for x in data:
    dict1[x] += 1
print(dict1)


# 用collections.Counter对象来完成上述的计数的情况

from collections import Counter

dict2 = Counter(data)
print(dict2)
most = dict2.most_common(3)
print(most)
print(type(most))

# 对一个文本进行词频统计
import re
txt = "da dd d s s s s s  s   s dfdf  fd fdf dd  dd  d  dd"
c3 = Counter(re.split('\W+', txt))
print(c3.most_common(3))
print(c3)


