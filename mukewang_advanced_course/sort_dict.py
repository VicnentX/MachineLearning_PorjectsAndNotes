# 根据字典里面的值的小大 对字典的项进行排序
from random import randint
# method1 内置函数 内部以c的速度运行 sorted
data = sorted([4,3,5,6,8,2,1,3])
print(data)
d = {x: randint(60, 100) for x in 'abcde'}
print(d)
list = sorted(d)
print(list)
# 用zip函数 将字典元祖话（value， key） 然后排序
d.keys()
d.values()
list_valueKey = zip(d.values(), d.keys())
print(list_valueKey)
sorted_list = sorted(list_valueKey)
print(sorted_list)

# method2 用items
list3 = sorted(d.items(), key=lambda x: x[1])
print(list3)
