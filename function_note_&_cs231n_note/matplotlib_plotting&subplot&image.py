"""

Matplotlib
Matplotlib is a plotting library.
In this section give a brief introduction
to the matplotlib.pyplot module,
which provides a plotting system similar to that of MATLAB.
"""

import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
print("x's shape is ", x.shape)
print("x's size is ", x.size)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.


print("With just a little bit of extra work "
      "we can easily plot multiple lines at once, "
      "and add a title, legend, and axis labels:")
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


print("Subplots "
      "You can plot different things in the same figure "
      "using the subplot function. Here is an example:")
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(3, 2, 1)    # 三个参数第一个是有几行 第二个有几列 最后是画第几个

# Make the first plot
plt.plot(x, y_cos)
plt.title('Cosine')

# Set the second subplot as active, and make the second plot.
plt.subplot(3, 2, 2)
plt.plot(x, y_sin)
plt.title('Sine')

# Show the figure.
plt.show()


print("Images___"
      "You can use the imshow function to show images. Here is an example:")

from scipy.misc import imread, imresize
import imageio

img = imageio.imread('cat.jpg')
img_tinted = img * [1, 0.6, 0.6]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()