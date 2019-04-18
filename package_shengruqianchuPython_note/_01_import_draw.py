# import antigravity
import turtle

print("---print stars---")
turtle.forward(100)
turtle.right(144)
turtle.forward(100)
turtle.right(144)
turtle.forward(100)
turtle.right(144)
turtle.forward(100)
turtle.right(144)
turtle.forward(100)
# turtle.done()

print("---彩色螺旋线---")

turtle.pensize(2)
turtle.bgcolor("black")
colors = ["red", "yellow", "purple", "blue"]
turtle.tracer(False)
for x in range(400):
    turtle.forward(2*x)
    turtle.color(colors[x % 4])
    turtle.left(91)
turtle.tracer(True)
turtle.done()
