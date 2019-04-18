# empty set
s = set()

s = set([1, 1, 3, 4, 4])
print(s)

s = set("it is a set")
print(s)

s.add(2)
print(s)


s.update("hello")
print(s)

s.remove(2)
print(s)

a = s.pop()
print(a)
print(s)

print(s - set("qwerit"))
print(s & set("qwerit"))
print(s | set("qwerit"))
print(" " in s)
print("ss" in s)
print("s" in s)
print(set("ab") | set("bc"))
