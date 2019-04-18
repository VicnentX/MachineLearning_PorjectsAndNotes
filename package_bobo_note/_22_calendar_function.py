# 根据公式获取星期几（1，2月当做上一年的12，13月）
# (d + 2 * m + 3 * (m + 1) // 5 + y + y // 4 - y // 100 + y // 400) % 7 + 1


def get_weekday_with_date(y, m, d):
    y = y - 1 if m == 1 or m == 2 else y
    m = 13 if m == 1 else (14 if m == 2 else m)
    w = (d + 2 * m + 3 * (m + 1) // 5 + y + y // 4 - y // 100 + y // 400) % 7 + 1
    return w


def get_days_in_month(y, m):
    if m in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif m in [4, 6, 9, 11]:
        return 30
    elif is_leap_year(y):
        return 29
    else:
        return 28


def is_leap_year(y):
    if y % 400 == 0 or (y % 4 == 0 and y % 100 != 0):
        return True
    return False




# year, month = 2019, 2
year = int(input("---please input year---"))
month = int(input("---please input month---"))
days = get_days_in_month(year, month)


print("w1 w2 w3 w4 w5 w6 w7")
print("___________________")
for i in range(1, days + 1):
    w = get_weekday_with_date(year, month, i)
    if i == 1:
        print(f"{' ' * (w - 1) * 3}", end="")
    elif w == 1:
        print()
    print(f"{i:2d}", end=" ")
print("")
