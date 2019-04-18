import turtle

for i in range(5):
    turtle.forward(200)
    turtle.right(144)
turtle.done()


age = input("---input year---")
age_trip = age.strip()
print(age_trip)
age = int(age)
print(age)
if age > 18:
    print("adult")
else:
    print("child")


digit = input("---input number---")
if digit.isdigit():
    print("ok")
else:
    print("wrong input")