# list comprehensions
list = [x * x for x in range(10)]
print(list)

list = [x * x for x in range(10) if x % 2 == 0]
print(list)


dict = {"a": 1, "b": 2}
a = [k + ":" + str(v) for k, v in dict.items()]
print(a)


dict = {i: i ** 3 for i in range(5)}
print(dict)


set = {i * i for i in range(5) if i > 2}
print(set)


tup = (i for i in range(5))
print(tup)
print(type(tup))