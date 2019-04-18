# a, b, c = (1, 2, 3, 4, 5, 6)  wrong


print("---元组解包之后 "
      "若有多个接受值 并且有一个是＊c 赋值之后是一个list 不是tuple---")
a, b, *c, d = (1, 2, 3, 4, 5, 6)
e = (1, 2, 3, 5)    #d is still tuple

print(a)
print(b)
print(c)
print(d)
print(e)



print("----dict是无序容器 要当心！！！----")
print("---dict解包 什么都不写 接的是key 不是value---")
m, *n = {"a": 1, "b": 2, "c": 3}
print(m)
print(n)

print("---dict解包  写。values（） 是接value---")
m, *n = {"a": 1, "b": 2, "c": 3}.values()
print(m)
print(n)

print("---dict解包  写。items（） 接的就是所有 把每个简直对包装成一个元组")
m, *n = {"a": 1, "b": 2, "c": 3}.items()
print(m)
print(n)


print("元组解包的反向使用,*l表示把元组解包")


def f6(*args):
      print(args)


l6 = [1, 2, 3, 4, 5, 6, 7, 8]
f6(*l6)


print("传递dict类型的参数解包时候必须是＊＊l")


def f7(**kwargs):
      print(kwargs)


l7 = {"a": 1, "b": 2, "c": 3}
f7(**l7)
