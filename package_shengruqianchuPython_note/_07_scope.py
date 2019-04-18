# L (local) 局部作用域
# E (Enclosing) 闭包函数外的函数中
# G (global)
# B (built-in) 内建作用域

"""
x = int(2.9)    # B
global_var = 0  # G


def outer():
    out_var = 1 # E
    def inner():
        inner_var = 2   # L
"""

# 查找顺序 L - E - G - B
# 以下函数会先从
# 局部函数变量，闭包外函数变量， 全局变量， 内建变量
# 的顺序来找那个a

a = 1


def func():
    print(a)


func()


# nonlocal - 引用闭包外面的变量
# 而不是全局变量 当然也不是内部定义的变量

#ex.1
name = "jack"


def outer():
    name = "peter"
    def inner():
        name = "mary"
        print(name)
    inner()

# output mary
outer()

# ex.2
name = "jack"


def f1():
    print(name)


def f2():
    name = "eric"
    f1()


# output jack
# !!!!这里有个很tricky的地方 往外找是def f1的外面 而不是f1()的外面
f2()



# ex.3 output jack
name = "jack"


def f2():
    name = "eric"
    return f1


def f1():
    print(name)


ret = f2()
ret()
