# dict : {key1: value1, key2: value2}
# key必须是不可变类型,list不能作为key

d = {"Beijing": 22, "Shanghai": 24}

print("-----search-----")
print(d["Beijing"])
print(d.get("Beijing"))
print(d.setdefault("Beijing"))

print("-----edit-----")
d["Beijing"] = 26
print(d)

print("-----add-----")
d["Guangzhou"] = 21
print(d)
d.update({"Hangzhou": 12})
print(d)

print("----get length-----")
print(len(d))
print("----convert dict to a string-----")
print(str(d))

print("-----in or not in-------")
print("Hangzhou" in d)
print("Xiamen" in d)

print("-----delte-----")
d.pop("Beijing")
print(d)
d.clear()
print(d)


d = {"price": 19.9, "name": "sprint", "count": 2, "volume": 500}
d["price"] = 9.9
d["made in"] = "China"
print(d)

print("----add multipile elements----")
d.update({"caroli": 120, "company": "cocacola"})
print(d)

print("-----delte : pop() use the value of key as its return value-----")
v = d.pop("price")
print(v)
print(d)

print("----pop(" ", -1)----")
v = d.pop("price", -1)
print(v)
print(d)


print("-----search-----")
v = d.get("price", -1)
print(v)

print("----setdefault(" ", number)-----")
'''
举个例 
d.setdefault("AA" , 7)
如果d里面有"AA" 那么就返回value的值
没有的话 就做2件事情 返回7 并且把"AA"： 7 放进d里面
'''


d.clear()
print(d)

print("---------------for dict--------------")
d1 = {"xiaowang": 85, "xiaohong": 85, "xiaoming": 78}
for i in d1.values():
    print(i, end=" ")
print()
for i in d1.keys():
    print(i, end=" ")
print()
for k, v in d1.items():
    print(str(k), str(v), end=" ")
print()

l = list(d1.keys())
print(l)
l = list(d1.values())
print(l)
l = list(d1.items())
print(l)

print("--------------exercise dict--------------")
s = dict(a=1, b=2, c=3)
for i in s:
    print(f"key = {i}, value = {s[i]}")
for k, v in s.items():
    print(f"key = {k}, value = {v}")

if 3 in s.values():
    print("Y")
if 5 in s.values():
    print("Y")

print(type(s.values()))
print(type(s.keys()))
