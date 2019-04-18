"""\
文件读写
file－like 类似文件对象

f = open("filename", "mode")
pass
f.close()


1.
filename － r"d:\2\test.py"

2.
mode 有以下几种
r 只读模式 如果文件不存在就报错 存在就正常读取
w 只写模式 如果文件不存在就新建文件再写 存在就先清空文件再写
a 追加模式 如果文件不存在 新建 写入， 存在就在文件最后追加写入
x 新建模式 比w安全 文件存在就报错
b 二进制模式 比如rb wb ab ，以bytes类型操作数据
+ 读写模式 比如r+ w+ a+ 既能写又能读 不推荐用这种模式

"""


# f = open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt", "w")
# f.write("I like python")
# f.write("\n")
# f.write("python is good")
# f.close()
#
# f = open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt", "r")
# data = f.read()
# f.close()
# print(data)
# datas = data.split("\n")
# print(type(data))
# print(type(datas))


# 按行读取
# f = open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt")
# list = f.readlines()
# print(list)
# print(len(list))
# f.close()
#
# f = open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt", "w")
# list[3] = "some changes"
# for line in list:
#     f.write(line)
# f.close()
#
# print(len(list))


#遍历文件
# f = open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt", "a")
#
# f.write("haha\n")
# f.write("xixi\n")
#
# f.close()

#tell() 以字节为单位计算
# seek()：改变指针位置 以字符为单位计算
# f.seek(offset, from_what)
# 0 means from beginning of file, 1 means from 当前位置, 2 means from the end of file

# f = open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt")
# pos = f.tell()  # 告诉我当前位置
# print(pos)
# a = f.read(16)  # 读取16个字符
# print(a)
# pos2 = f.tell()    # 一个中文字符占3个字节，在gb2312下占两个字节
# print(pos2)
# f.close()

# with关键词用于python的上下文管理器机制

with open(r"/Users/Vincent_Xia/Downloads/read_write_test.txt") as f:
    data = f.read()

# 不用写close 会自动帮我们close ， with支撑打开多个文件 open() as f1, open() as f2:







