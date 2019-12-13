d = {}
d['leo'] = (2, 37)
d['Bob'] = (3, 40)
d['jim'] = (1, 35)


for k in d:
    print(k)


from time import time
from random import randint
from collections import OrderedDict

d = {}
players = list('ABCD')
start = time()

for i in range(4):
    input()
    p = players.pop(randint(0, 4 - i))
    end = time()
    print(i + 1, p, end - start)
    d[p] = (i + 1, end - start)

print(d)