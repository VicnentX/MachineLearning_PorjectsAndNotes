# while is used in: 不知道具体循环次数，但知道循环结束的条件

# break continue

l = []
while True:
    i = input(r"---input your plan and 'q' means quit---")
    if i == "q":
        break
    l.append(i)
print(l)

# pass means doing nothing
# python does not allow empty statement

for i in range(5):
    pass
