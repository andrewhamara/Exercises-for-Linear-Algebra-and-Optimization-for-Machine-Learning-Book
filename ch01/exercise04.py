# problem: Show that if we have a matrix satisfying A = -A.T,
#          then for any column vector x, we have x.T * Ax = 0.



import numpy as np

# Create a matrix satisfying A = -A.T
A = np.ones((3,3))
x = np.zeros(3)
np.fill_diagonal(A, x)
lower_triangle_indices = np.tril_indices(A.shape[0], k=-1)
A[lower_triangle_indices] = -1

# Get a random column from A
column_index = np.random.randint(0,3)
x = A[:, column_index]

# Use A to transform that column vector
rhs = np.dot(A, x)

# We don't need to transpose x because NumPy is nice, but we do anyways
result = np.dot(x.T, rhs)

# Approximately zero
assert -1e-9 < result < 1e-9