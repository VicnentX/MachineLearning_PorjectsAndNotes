# [表达式 for 变量 in 列表]
# .upper()
# print("---print all element in the console---")
# l8 = [1, 3, 5, 7, 9]
# [print(i) for i in l8]


l1 = [1, 2, 3, 4, 5]
l2 = [i * 2 for i in l1]
print(l1)
print(l2)

l3 = [i // 2 for i in range(10) if i % 2 == 0]
print(l3)

l4 = [i for i in "hello"]
print(l4)

l5 = [i.upper() for i in "hello" if i != "o"]
print(l5)

print("---get initial character and form a new list---")
l6 = ["Food", "Moon", "Loop"]
l66 = [i[0] for i in l6]
print(l6)
s = "".join(l66)
print(s)


print("----all the elements that occurs in l7 and l77---")
l1 = [2, 4, 6, 8, 10, 12]
l2 = [3, 6, 9, 12]
r2 = [i for i in l1 if i in l2]
print(r2)


print("---print all element in the console---")
l8 = [1, 3, 5, 7, 9]
[print(i) for i in l8]


print ("---2 loops----")
l9 = [[j for j in range(5)] for i in range(3)]
print(l9)

l10 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
l1010 = [[print(j) for j in i] for i in l10]
print (l1010)


