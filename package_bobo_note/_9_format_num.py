PI = 3.1415926
result = f"PI is {PI}"
print(result)
result = f"PI is {PI:.2f}"  # 有进位
print(result)
result = f"PI is {PI:50.2f}"    # 50 is wide
print(result)

tem = 3.1415
result = f"PI is {tem:+50.2f}"  # + means show sign whenever it is + or -
print(result)
result = f"PI is {tem:<+50.2f}"  # + means show sign whenever it is + or -
print(result)
result = f"PI is {tem:^+50.2f}"  # + means show sign whenever it is + or -
print(result)
result = f"PI is {tem:*^+50.2f}"  # + means show sign whenever it is + or -
print(result)
result = f"PI is {tem:.0%}"  # + means show sign whenever it is + or -
print(result)

print('------转进制-----')
'''
b - 2
o - 8
x - 16
#b - start with 0b
#o - start with 0o
#x - start with 0x'''
tem = 25
result = f"PI is {tem:#b}"  # + means show sign whenever it is + or -
print(result)
result = f"PI is {tem:x}"  # + means show sign whenever it is + or -
print(result)




