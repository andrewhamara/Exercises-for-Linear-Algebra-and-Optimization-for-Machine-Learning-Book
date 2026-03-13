# problem: For any two vectors x and y, which are each of length a, show that:
#     (i): x - y is orthogonal to x + y,
#     (ii): the dot product of x - 3y and x + 3y is negative

import numpy as np

# two random 100-length column vectors
x = np.random.rand(100)
y = np.random.rand(100)

# normalize to unit vectors
x_normalized = x / np.sqrt(np.dot(x,x))
y_normalized = y / np.sqrt(np.dot(y,y))

# first, let's do part (i) and show that the vectors (x - y) and (x + y) are orthogonal:

# let's use the normalized version, since we know they have the same length
a = x_normalized + y_normalized
b = x_normalized - y_normalized

result = np.dot(a,b)

# if (x - y) and (x + y) have a dot product of 0, we know they are orthogonal.
# Since we're using a computer, we should approximate:
assert -1e-9 < result < 1e-9

# now let's do part (ii), showing that (x - 3y) and (x + 3y) have a negative dot product.

# let's define these vectors:
v1 = x - 3*y
v2 = x + 3*y

# and check the result
result = np.dot(v1, v2)
assert result < 0