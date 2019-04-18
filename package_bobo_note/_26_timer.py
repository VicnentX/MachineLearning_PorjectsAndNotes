import datetime


# 把函数当做参数 前面要用if 判断 是不是None


def timer(t, start, finished):
    t_start = datetime.datetime.now()
    print("start")
    if start:
        start()
    while True:
        t_end = datetime.datetime.now()
        delta_t = t_end - t_start
        if delta_t.seconds >= t:
            break
    #print("end")
    if finished:
        finished()


def f():
    i = int(input("---please input a number---"))
    print(i)


def s():
    print("---please wait for 2 sec")


timer(2, s, f)

