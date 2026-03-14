# problem: show that the matrix product AB remains unchanged
#          if we scale the ith column of A and the jth column
#          of B by respective factors that are inverses of each other.

# Before the code, which is pretty simple, how is it that this is the case?
# I recommend drawing out two 3x3 matrices of ones and seeing that it will
# result in another 3x3 matrix will all threes. Then, change one column of A
# with some other value, I did 4, and a row of B with 1/n (1/4 in my case).
# It will be clear to see that you'll only ever multiple your scaled values
# by each other, so they're counteracting one another.

import numpy as np

A = np.random.rand(3,3)
B = np.random.rand(3,3)

result = np.matmul(A, B)

# random int in {0,1,2}
col_and_row = np.random.randint(0,3)

# random scaling factor
scale = np.random.randint(0, 1_000)

# multiply one by the scale and divide the other
A[:, col_and_row] *= scale
B[col_and_row] /= scale

result2 = np.matmul(A, B)

# the results before and after the scaling are equivalent
assert result.all() == result2.all()