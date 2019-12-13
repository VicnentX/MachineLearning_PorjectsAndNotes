from random import randint

import v as v

data = [randint(-10, 10) for _ in range(10)]

print(data)

# 列表解析最快
data1 = [x for x in data if x >= 0]
print(data1)

# filter次之
print(filter(lambda x: x >= 0, data))
print(list(filter(lambda x: x >= 0, data)))

# for loop 最慢

# ------------------------------

data_dict = {x: randint(60, 100) for x in range(1, 21)}
print(data_dict)

data1 = {k: v for k, v in data_dict.items() if v > 90}
print(data1)

# ------------------------------
# 集合set 解析
s = set(data)
data1 = {x for x in s if x > 4}
print(data1)
