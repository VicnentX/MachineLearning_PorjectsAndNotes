'''
if conditionA:
    statement


if conA
    statement1
else
    statement2
'''


num = input("input a number : ")
num = int(num)

if num % 2 == 0:
    print(f"you input is {num}, it is a even")
    if num % 3 == 0:
        print(f"{num} could be divided by 3")
else:
    print(f"you input is {num}, it is an odd")


'''
if A:
    state1
elif B:
    state2
elif C:
    state3
else:
    state4
'''

score = input("input your score: ")

# "" , 0 , None , False : all these 4's bool is false
if score : # equals score != ""
    score = int(score)
    if 0 <= score <= 100 :
        if score == 100:
            print("s")
        elif 90 <= score:
            print("A")
        elif 80 <= score:
            print("B")
        elif 70 <= score:
            print("C")
        elif 60 <= score:
            print("D")
        else:
            print("E")
    else:
        print("wrong score")
else:
    print("no input")



