def f(x):
    return x * 2


# 把上面设计一个匿名函数如下：


lambda x: x * 2


# lambda 不能换行

a = lambda x: x * 3
b = lambda x, y: x + y # 不能使用for while 和 赋值

print(a(5))
print(type(a))
print(b(1, 6))



