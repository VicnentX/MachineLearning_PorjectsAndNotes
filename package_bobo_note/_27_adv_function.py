def f1():
    def f2():
        print("this is f2 fucntion")
    return f2


a = f1()
a()

# bi bao


def counting(count):
    def fn():
        nonlocal count
        count -= 1
        if count < 0:
            return
        return count
    return fn


# 两个不影响的计数器
# 每个bi bao 会一起把函数和执行函数的语句（或者说参数 比如count）一起打包
a = counting(5)
b = counting(3)
print('a', a())
print('a', a())
print('a', a())
print('a', a())
print('a', a())
print('a', a())
print('b', b())
print('b', b())
print('b', b())
print('b', b())
