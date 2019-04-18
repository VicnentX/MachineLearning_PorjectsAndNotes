# 变量作用域
# 局部变量local  在函数内定义的变量
# 全局变量global  在函数外定义的变量
# global var 可以在函数内部使用 ， 不能在函数内部直接修改
# 函数内可以定义一个和函数外 名字一样 的变量

# 如果要在函数内修改全局变量的值 需要使用global关键字

print("---example - NOT change global var---")
i = 5


def f():
    i = 0


f()
print(i)

print("---example -  change global var---")
i = 5


def f():
    '''
    global var could be used with statement "global i"
    :return:
    '''
    global i
    i = 1


f()
print(i)
print(f.__doc__)

