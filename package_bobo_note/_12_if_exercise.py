import random

year = input("---input a year---")
year = int(year)
if year % 400 == 0 or year % 4 == 0 and year % 100 != 0:
    print(f"{year} is run nian")
else:
    print(f"{year} is NOT run nian")


number = input("input a number between 1 -99999:")
number = int(number)
a1 = number % 10
a10 = number // 10 % 10
a100 = number // 100 % 10
a1000 = number // 1000 % 10
a10000 = number // 10000 % 10

print(f"ge{a1} + shi{a10} + bai{a100} + qian{a1000} + wan{a10000}")


'''
1 - jiandao
2 - shitou
3  - bu'''
number = random.randint(1, 3)
user_number = input("input a number")
user_number = int(user_number)

if number == user_number:
    print("tie")
else:
    if user_number > number and not (user_number == 3 and number == 1) or user_number == 1 and number == 3:
        print("you win")
    else:
        print("you lose")

print(str(user_number) + ',' + str(number))