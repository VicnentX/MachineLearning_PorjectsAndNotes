# 递归 － 在函数内部调用本身
# python 最多1000次递归


def f(i):
    print(i)
    if i <= 1:
        return 1
    else:
        return f(i - 1) * i


r = f(5)
print(r)


def func(num):
    ret = 1
    for i in range(num):
        ret *= i + 1
    return ret


print(func(5))


# lambda is NOT recommended
f2 = lambda i: 1 if i <= 1 else f2(i - 1) * i
r2 = f2(5)
print(r2)
