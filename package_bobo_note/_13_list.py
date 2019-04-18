'''
method
a.sort()
a.reverse()

function
len(a)
min(a)
max(a)
'''



a = [] # empty list
b = [1, 3, 5, 7]
print(type(b))
b = [1, 3.14, "H", False, 4, 5.64]
print(b[1])
print(b[-1])

b[0] = "Hi"
print(b)
print(b[1:4])
print(b[1::2])
print(b[::-1])
print("------two ways-----")
print(b[3:0:-1])
tem = b[3:0:-1]
print(tem)

print("----operation to list")

my_list = [1, 2, "hi", 5]
print(len(my_list))
print(2 in my_list)
print(my_list * 2)
my_list.reverse()   # 这个没有返回值
print(my_list)

# 以下三种方法在int 和 string 混编的list里面不能用 会报错
# max_value = max(my_list)
# min_value = min(my_list)
# my_list.sort()
# print(my_list)


print("----new exercise---")
a = [1, 2, 3, 4, 5, 6, 7]
b = ["a", "b", "c"]
if 5 in a:
    print("5 is in a")
else:
    print("5 is not in a")
n = len(a)
print(n)
print(a[n - 1])
new_list = a + b * 2
print(new_list)
a.reverse()
print(f"a = {a}")
print(max(a))
print(min(a))


print("-------add,delete,edit,search-------")
print("-------edit--------")
c = [1, 2, 3, 4, 5, 5, 5]
c[1] = 0
print(c.index(5))  # 不在的话 会报错
print(c.count(5))
print(c)

d = ["tue", "mon", "wed"]
d[0], d[1] = d[1], d[0]
print(d)

e = "mon,tue,wed"
tem = e.split(",")
print(tem)

print("-----list to string-----")
f = "*".join(tem)     # tem must be string list
print(f)
f = " ".join(tem)     # tem must be string list
print(f)


print("-------add--------")
g = [1, 2, 3, 4]
g.append(5)
print(g)

g.insert(2, 2.5)
print(g)
print(type(g))
g.insert(100, 8)
print(g)
g.insert(-100, 0)
print(g)

# 比如index －1 那么插入到最后一位之前
g.insert(-1, 7)
print(g)
g.extend([9, 10, 12])
print(g)

s = "hello"
g.extend(s)
print(g)
g.extend([s])
print(g)


print("-----delete------")
print("---pop return value is the element"
      "and "
      "cannot out of range"
      "but can use -1-2-3 ----")
print("---del must use index"
      "and "
      " has no return value")
print("---remove method make sure that element is in list")
h = [1, 3, 5, 7]
h.pop()
print(h)
h.pop(0)
print(h)
del h[0]
print(h)
h.append(7)
print(h)
h.remove(5) # 5 is the element not index
print(h)
h.clear()
print(h)

print("----nested list-----")
stu1 = ["eric", 95]
stu2 = ["maddee", 85]
stu3 = ["carolina", 88]
students = [stu1, stu2, stu3]
print(students)
students.remove(stu3)
print(students)
print(students[0])
print(students[0][1])


print("-----list exercise----")
j = [30, 31] * 5
j.pop(0)
j[1] = 28
j.insert(7, 31)
j.extend([30, 31])
print(j)

print("----input a month and output how many day in this month")
month = input(" input a month: ")
print(f"{month}th month has {j[int(month) - 1]} days")





