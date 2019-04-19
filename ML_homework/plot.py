import numpy as np
import pylab as pl


"""
2.1.1 折线图 Line plots(关联一组x和y值的直线)
"""
# make an array of x values
x = [1, 2, 3, 4, 5]

# make an arrat of y values for each x value
y = [1, 4, 9, 16, 25]

# use pylab to plot x and y
pl.plot(x, y)
# show the plot on the screen
pl.show()


"""
2.1.2 散点图 Scatter plots
o 代表 只画点

b blue
g green
r red
c cyan
m magenta
y yellow
k black
w white
"""

pl.plot(x, y, 'o')
pl.show()
pl.plot(x, y, 'or')
pl.show()

"""
2.2.2 线条样式 Changing the line style
"""

pl.plot(x, y, "r--")
pl.show()
pl.plot(x, y, "r*")
pl.show()

"""
s square marker
p pentagon marker
* star marker
h hexagon1 marker
H hexagon2 marker
+ plus marker
x x marker
D diamond marker
d thin diamond marker
"""
pl.plot(x, y, "ys")
pl.show()


"""
2.2.4 图和轴标题以及轴坐标限度 Plot and axis titles and limits
"""
x = [1, 2, 3, 4, 5]  # Make an array of x values
y = [1, 4, 9, 16, 25]  # Make an array of y values for each x value
pl.plot(x, y)  # use pylab to plot x and y
pl.title("Plot of y vs.x")  # give plot a title
pl.xlabel("x axis")  # make axis labels
pl.ylabel("y axis")

pl.xlim(0.0, 7.0)  # set axis limits
pl.ylim(0.0, 30.)

pl.show()  # show the plot on the screen


"""
2.2.6  图例 Figure legends

pl.legend((plot1, plot2), (’label1, label2’), 'best’, numpoints=1)

其中第三个参数表示图例放置的位置:'best’‘upper right’, ‘upper left’, ‘center’, ‘lower left’, ‘lower right’.

如果在当前figure里plot的时候已经指定了label，如plt.plot(x,z,label="cos(x2)")，直接调用plt.legend()就可以了哦。
"""


x1 = [1, 2, 3, 4, 5]    # Make x, y arrays for each graph
y1 = [1, 4, 9, 16, 25]
x2 = [1, 2, 4, 6, 8]
y2 = [2, 4, 8, 12, 16]


# , 不能少
plot1, = pl.plot(x1, y1, "r")  # use pylab to plot x and y : Give your plots names
plot2, = pl.plot(x2, y2, "go")

pl.title("Plot of y vs.x")  # give plot a title
pl.xlabel("x axis")  # make axis labels
pl.ylabel("y axis")


pl.xlim(0.0, 9.0)  # set axis limits
pl.ylim(0.0, 30.)

pl.legend([plot1, plot2], ["red line", "green circles"], loc="best")  # make legend
pl.show()  # show the plot on the screen
