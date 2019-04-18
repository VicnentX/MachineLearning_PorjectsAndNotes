# the difference between list and tuple :
# tuple is constant type
# tuple 可以作为函数的参数或者返回值来使用
# 系统处理tuple更快
# tuple 可以作为dict的key list不能做dict的key

a = (1, 2, 3)
print(type(a))
print(a)
b = (1)
print(type(b))
b = (1,)
print(type(b))
a = 1, 2, 3
print(type(a))
print(a + b)
print(a * 3)
print(a.count(1))
print(a.index(3))

# 关联性强的情况下用tuple

position = (12, 10)
color = (255, 255, 255)


t = 1, 2, 3
aa, bb, cc = t
print(f"t = {t} aa = {aa}, bb = {bb}, cc = {cc} ")




