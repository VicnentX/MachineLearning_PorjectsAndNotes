print("---默认参数 使用方法:"
      "参数设置默认值只能从后往前---")


def f(name, age = 0):
    print(f"name = {name} , age = {age}")


f("xiaoming", 23)
f("xiaohong")


print("---关键词参数 例子 当用了关键词参数，后面必须都用关键词参数---")


def fabc(a, b=4, c=9):
    print(a, b, c)


fabc(a=1, b=2, c=3)
fabc(a=1, c=3, b=2)
print("haha", "123", sep="* ")
fabc(12, c=5, b=4)
# fabc(12, a=5, b=3) 这个是错的
# fabc(12, c=5, 4) wrong
fabc(12)
fabc(12, 6)
fabc(12, c=6)


l = [1, 2, 3]
a = l
a.append(6)
print(l)
a = [8, 7, 2]
a.append(3)
print(a)


print("---可变参数，就是可以传递任意数量的参数值 数量类型都没有限制---")
print("---可变参数单独使用 不支持关键词赋值---")
print("---可变参数 只能有一个---")

def get_sum(*args):
    sum = 0
    for i in args:
        sum += i
    print(sum)


get_sum(1, 2, 5)    #得到一个tuple


def f3(a, b, *args):
    print(a)
    print(b)
    print(args)


f3(1, 2, {3: 5, 5: 6})    #得到一个tuple

print("---不可变参数和可变参数一起使用时候，如下"
      "调用函数时候，可变参数之前，只能使用顺序赋值，"
      "调用函数时候，可变参数之后，只能使用关键词赋值")


print("---args会以元组的形式保留所有参数---")
def f4(a, b, *args, c, d):
    print(a)
    print(b)
    print(args)
    print(c)
    print(d)


f4(1, 2, 3, 4, 5, 6, c=7, d=8)


print("--- **kwargs 只能出现在参数列表中最后的位置 ---")
print("---一般就是：一般参数，可变参数，可变关键词参数---")

def f5(a, b, c, *args, d, **kwargs):
    print(args)
    for k, v in kwargs.items():
        print(f"key = {k} , value = {v}")


f5(1, 2, 3, 4, 5, 6, x='hi', d=55, y='hello')

