#循环中遍历用到range

# range(end) : 0 - end-1
l3 = list(range(100))
print(l3)

# range(start, end): start - end-1
l4 = list(range(4, 12))
print(l4)

# range(start, end, step): start - end-1 with step
l5 = list(range(3, 9, 2))
print(l5)

l6 = tuple(range(3, 9, 2))
print(l6)
