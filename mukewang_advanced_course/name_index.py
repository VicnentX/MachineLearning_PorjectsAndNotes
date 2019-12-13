# method1 列表拆包
NAME, AGE, GENDER, EMAIL = range(4)
student = ('Jim', 16, 'Male', 'Jim82@gmail.com')
print(student[NAME])
print(student[GENDER])

# method1 collections.namedtuple replace 内置tuple
from collections import namedtuple
Student = namedtuple('StudentTemple', ['NAME', 'AGE'])
s1 = Student('Jim', 16)
print(s1)
print(type(s1))

s2 = Student(NAME='Jim', AGE=16)
print(s2)

print(s2.NAME)
print(s2.AGE)

# s2 所属的类是tuple的一个子类
print(isinstance(s2, tuple))






