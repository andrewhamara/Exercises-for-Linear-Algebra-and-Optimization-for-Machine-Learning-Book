# problem: Consider a situation in which you have three matrices A, B, and C,
# of sizes 10x2, 2x10 and 10x10, respectively

# (a) Suppose you had to compute the matrix product ABC. From an efficiency
# perspective, would it computationally make more sense to compute (AB)C
# or would it make more sense to compute A(BC)?

# (b) If you had to compute the matrix product CAB, would it make more sense
# to compute (CA)B or C(AB)?

import numpy as np

# Here, what we want to find out is how many multiplications will occur.
# We can do this by ignoring the entries of the matrices, just creating them:

A = np.ones((10,2))
B = np.ones((2, 10))
C = np.ones((10, 10))

# Let's start with part (a):

# AB(C)
m_1 = 0
m_1 += A.shape[0] * A.shape[1] * B.shape[1]
resulting_matrix_1 = np.matmul(A, B)
m_1 += resulting_matrix_1.shape[0] * resulting_matrix_1.shape[1] * C.shape[1]
print(f'(AB)C had {m_1} multiplications') # -> 1200

# A(BC)
m_2 = 0
m_2 += B.shape[0] * B.shape[1] * C.shape[1]
resulting_matrix_2 = np.matmul(B, C)
m_2 += A.shape[0] * A.shape[1] * resulting_matrix_2.shape[1]
print(f'A(BC) had {m_2} multiplications') # -> 400

# we prefer A(BC)
assert m_1 > m_2

################################################################################
# let's move on to part (b) now

# here, we're doing CAB, but it's the same work as before:
# do we want (CA)B or C(AB)?

# CA(B)
m_3 = 0
m_3 += C.shape[0] * C.shape[1] * A.shape[1]
resulting_matrix_3 = np.matmul(C, A)
m_3 += resulting_matrix_3.shape[0] * resulting_matrix_3.shape[1] * B.shape[1]
print(f'(CA)B had {m_3} multiplications') # -> 400

# C(AB)
m_4 = 0
m_4 += A.shape[0] * A.shape[1] * B.shape[1]
resulting_matrix_4 = np.matmul(A, B)
m_4 += C.shape[0] * C.shape[1] * resulting_matrix_4.shape[1]
print(f'C(AB) had {m_4} multiplications') # -> 1200

# we prefer CA(B)
assert m_4 > m_3