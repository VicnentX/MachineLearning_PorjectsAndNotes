# for ... in... : could be used in tuple, list, range()


l = list(range(6))
for item in l:
    item += 2
    print(item)
print(l)


for item in "python":
    print(item)

for item in range(3):
    print(item)

for _ in range(10):     # _ means this item not used in the loop
    print("hello world!")

sum = 0
for item in range(1, 101):
    sum += item
print(sum)

sum = 0
for i in range(100):
    if i % 2 == 0:
        sum += i
print(sum)


nums = input("----input 5 numbers with ,---")
list = nums.split(",")

sum = 0
for i in range(5):
    list[i] = int(list[i]) * 2
    sum += list[i]
print(sum)
print(nums)

print("----find out all the numbers that is 7X or contains 7---")
list_of_7 = []
for i in range(100):
    if i % 7 == 0 or str(i).count("7") != 0:
        list_of_7.append(i)
print(list_of_7)

