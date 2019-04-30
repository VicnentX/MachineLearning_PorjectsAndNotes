import numpy as np

a = np.array([2, 3, 4])
print(a)
print(type(a))
a = np.arange(1, 12, 2)
print(a)
print(type(a))
a = np.linspace(1, 12, 6)
print(a)
print(type(a))
a = a.reshape(3, 2)
print(a)
print(type(a))
print(a.size)
print(a.shape)
print(a.dtype)
print(a.itemsize)   #one element 8 bytes
b = np.array([(1.5, 2, 3), (3, 4, 5)])
print(b)
print(type(b))
print(b.dtype)
print(a < 4)
print(a)
a *= 3
print(a)
print(type(a))
c = np.zeros((3, 4))
print(c)
print(type(c))
print(c.dtype)
print(np.ones(10))
print(np.ones((2, 3)))
print(np.array([2, 3, 5], dtype=np.int16))
r = np.random.random((2, 3))
print(r)
print((type(r)))
r = 3 * np.random.random((2, 3))
print(r)
print((type(r)))

np.set_printoptions(precision=4, suppress=True)
print(r)
print(np.random.random((2, 3)))

ir = np.random.randint(1, 100, 9)
print(ir)
sum_of_ir = ir.sum()
print(sum)
print(ir.min())
print(ir.max())
print(ir.mean())

np.set_printoptions(precision=4, suppress=True)
# 方差
print(ir.var())
# 标准差
print(ir.std())

print("_____________________")
a = np.random.randint(1, 100, 6)
a = a.reshape((2, 3))
print(a)
print(a.sum(axis=1))
print(a.sum(axis=0))
print(a.var())
print(a.var(axis=1))

print("---------从文件中导入数据－－－－－－－－－－")
# data = np.loadtxt("data.txt", dtype=np.uint8, delimiter=",", skiprows=1) 跳过第一行

print("____________打乱顺序————————————————————")
a = np.arange(100)
a = a.reshape((10, 10))
print(a)
np.random.shuffle(a)
print(a)
# random_choice = np.random.choice(a)
# print(random_choice)
lis = np.random.choice(5, 3)
print(lis)
print(lis.dtype)
print("______random choice only select from 1-d array______")
lis = np.random.choice(5, 3, p=[0.1, 0, 0, 0, 0.9])
print(lis)
print("-------select 2 int from 5 - 10--------")
lis = np.random.randint(5, 10, 2)
print(lis)
print("lis type", type(lis))

print("Numpy also provides many functions to create arrays:")
a = np.zeros((2, 2))
b = np.zeros((1, 2))
c = np.full((2, 3), 7)  # fill with 7.0
d = np.eye(6)   #Id matrix with rank 6
e = np.random.random((2, 5))

print("Array indexing\n"
      "Numpy offers several ways to index into arrays.\n"
      "Slicing: Similar to Python lists, numpy arrays can be sliced. "
      "Since arrays may be multidimensional, you must specify a slice for each dimension of the array:")

a = np.arange(1, 13)
a = a.reshape((3, 4))
b = a[:2, 1:3].copy()
print(b)
print(a[0, 1])
b[0, 0] = 77
print(b)
print(a[0, 1])

row_r1 = a[1, :]
row_r2 = a[1:2, :]
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

a = np.arange(1, 7)
a = a.reshape((3, 2))
print(a[[0, 1, 2], [0, 1, 0]])

b = np.array([a[0, 0], a[1, 1], a[2, 0]])
print(b)
print(b.shape)
print(a[[0, 0], [1, 1]])
print(a[[0, 0], [1, 1]].shape)


# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])

print("Boolean array indexing: "
      "Boolean array indexing lets you pick out arbitrary elements of an array. "
      "Frequently this type of indexing is "
      "used to select the elements of an array that satisfy some condition."
      " Here is an example:")

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"
print(bool_idx.shape)
# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"
print(a[bool_idx].shape)

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
print(a[a > 2].shape)


print("_______________Datatypes_________________")
x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"


print("_________________arraymath_______________")
# Basic mathematical functions operate elementwise on arrays,
# and are available both as operator overloads and as functions
# in the numpy module:

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

print("_____above Note that "
      "unlike MATLAB, * is elementwise multiplication, not matrix multiplication. "
      "We instead use the dot function to compute inner products of vectors, "
      "to multiply a vector by a matrix, and to multiply matrices. "
      "dot is available both as a function in the numpy module "
      "and as an instance method of array objects:")

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
z = np.array([[1,1],[2,2],[3,3]])

# print("x dot z ", x.dot(z))
print("x dot zT ", x.dot(z.T))

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

print("Numpy provides many useful functions for performing computations on arrays; one of the most useful is sum:")
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"


print("Apart from computing mathematical functions using arrays, "
      "we frequently need to reshape or otherwise manipulate data in arrays. "
      "The simplest example of this type of operation is transposing a matrix; "
      "to transpose a matrix, simply use the T attribute of an array object:")

x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.shape)
print(v.T)  # Prints "[1 2 3]"
print(v.T.shape)

v = np.array([[1,2,3],[4,5,6]])
vt = v.T
print(vt.shape)

vr = np.ravel(v, 0)  # 0 is by rows, 1 is by cols
print(vr)
v[0,0] = 9
print(v)
print(vr)
vrr = v.flatten()
print(vrr)

print("Broadcasting"
      "Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes "
      "when performing arithmetic operations. "
      "Frequently we have a smaller array and a larger array, "
      "and we want to use the smaller array multiple times to perform some operation on the larger array."
      "For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:")

x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
v = np.array([1,0,1])
y = np.empty_like(x)

m, n = x.shape

for i in range(m):
    y[i, :] = x[i, :] + v

print(y)

print("---------上面这种方式很慢，用下面这种方法快一些 但是！broadcasting更加吊－－－－－－－－－")
print("This works; however when the matrix x is very large, "
      "computing an explicit loop in Python could be slow. "
      "Note that adding the vector v to each row of the matrix x "
      "is equivalent to forming a matrix vv by stacking multiple copies"
      " of v vertically, then performing elementwise summation of "
      "x and vv. We could implement this approach like this:")

vv = np.tile(v, (m,1))
print(vv)
y = vv + x
print(y)

print("Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:")
y = v + x
print(y)

"""
The line y = x + v works even though x has shape (4, 3) and v has shape (3,) due to broadcasting; this line works as if v actually had shape (4, 3), where each row was a copy of v, and the sum was performed elementwise.

Broadcasting two arrays together follows these rules:

If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
The arrays can be broadcast together if they are compatible in all dimensions.
After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension
If this explanation does not make sense, try reading the explanation from the documentation or this explanation.

Functions that support broadcasting are known as universal functions. You can find the list of all universal functions in the documentation.

Here are some applications of broadcasting:
"""

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print("x is ", x)
print("v is ", v)
print("x + v is ", x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print("x", x)
n = w.shape
print("n is ", n)
print("x's shape is ", m, n)
print("xT", x.transpose())
print("xT", x.T)

print((x.T + w).T)

print("# Another solution is to reshape w to be a column vector of shape (2, 1),"
      "we can then broadcast it directly against x to produce the sameoutput.")
print(w.shape)
n = w.size
wT = w.transpose()
print("wT is :", wT)
print(wT == w)
print(wT.shape)
print(n)
print(x + np.reshape(w, (w.size, 1)))


