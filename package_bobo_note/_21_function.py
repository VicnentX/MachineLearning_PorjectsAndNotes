'''
name = input("--dfs--") # 可以没有参数也可以
print(name) 可以提供多个参数
l = [1, 2, 3]
l.append(4) ＃必须提供参数
len(l)
l.clear()

input append has return value
if there is no return value , the return value is None

all of them are functions
there are () in all of them

'''

i = 1
print(i)
print("hello", "world", "!", sep="*-", end="|\n")
print("************")

# get the function name
print(len.__name__)


# define a new function
# 函数名加下划线+小写
# 函数头 － def 函数名（参数列表）
# !!!!!先定义 再调用！！！！顺序重要！！！！


def say_morning():
    # 函数可以调用其他函数
    say_hi()
    print("Good Morning")


def say_hi():
    print("Hello")


say_hi()

# !!!运行say morning不可以在say hi之前 因为这个时候say——hi在小笨笨上
# 但是并没有执行 所以不用担心
# 放在最后执行就没问题 因为当我执行say hi的时候 say hi已经在小笨本上面 了
# !!!!比如a调用b b调用a 那么就死循环了

say_morning()

print("---function line(len) -----")


def line(width):
    print("-" * width)


line(10)
line(20)

# char: str 后面就有自动补全的功能比如.split
# ！！！！文章注视写在行数里面第一行第一行''' '''
def draw_line(width, char: str):
    '''
    画一条线
    :param width:
    :param char:
    :return:
    '''
    print(char * width)


draw_line(5, "*&")


print("----定义有返回值的函数------")
def get_max(a, b):
    return b if b > a else a


i = get_max(1, 3)
print(i)


def set_age(age):
    if type(age) != int or age < 0:
        return None
    print(age)


set_age(-3)
set_age("afe")
set_age(89)


print("-----multi return value fucntion-----")
def func(a, b):
    return a + b, a - b


i, j = func(3, 4)
print(f"i = {i} and j = {j}")

print("两个返回值但是只有一个接收 那么这个接受的就会变成一个tuple 而不是list")
k = func(3, 4)
print(k)







print()

