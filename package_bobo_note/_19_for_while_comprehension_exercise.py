import math

# print(a, b, c, sep = "*", end = "")

print("a", "b", "c", sep="*", end="")
print("a", "b", "c", sep="*", end="\n")

print("---from 100 - 1000 find min number "
      "which meets the pattern like "
      "407 = 4 ^ 3 + 0 ^ 3 + 7 ^ 3---")
for i in range(100, 1001):
    sum = 0
    for j in str(i):
        j = int(j)
        sum += (j * j * j)
    if i == sum:
        print(f"shuixianhuashu : {i}")
        break

print("----all prime in 1 - 100---")
l1 = []
for i in range(2 , 101):
    is_prime = True
    for j in range(2, int(math.sqrt(i))):
        if i % j == 0:
            is_prime = False
            break
    if is_prime:
        l1.append(i)
print(l1)



print("----9 * 9---")
for i in range(1, 10):
    for j in range(1, 10):
        print(f"{i} * {j} = {i * j:2d} ", end=" ")
    print()


