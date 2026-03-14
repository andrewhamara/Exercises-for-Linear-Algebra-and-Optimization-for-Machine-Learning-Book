# problem: Show that if a matrix A satisfies A = -A.T,
#          then all the diagonal entries in A are zero.

import numpy as np

# Here, we know that A=0 will satisfy the given property,
# but to solve this problem we have to think of other cases
# where A = -A transpose. This is nice, and we can think about
# square matrices here. In the transpose operation, we can think
# about swapping rows/columns, but also about swapping individual
# entries with their coordinates reversed. So (3,4) swaps with (4,3),
# and so on. So what we want, is to have a number x in (3,4), and -x
# in (4,3), and after transposing and negating, we are back where we
# started for non-diagonal entries. However, the non-diagonal entries
# have nothing to "swap" with when transposing, so negating non-zero
# entries would have to change this matrix.

dim = 3
A = np.ones((dim,dim))
diagonal = np.zeros(dim)

np.fill_diagonal(A, diagonal)

print(A) # a matrix of ones with zeros down the diagonal

lower_triangle_indices = np.tril_indices(A.shape[0], k=-1)

A[lower_triangle_indices] = -1

print(A) # matrix with 0 down the diagonal, ones in the upper triangle, and negative ones in the lower triangle

# so this satisfies the property from the problem
assert np.equal(A.all(), -1 * A.T.all())

# now let's try changing the diagonal (we know this won't work)
diagonal = np.ones(dim)
np.fill_diagonal(A, diagonal)

# and we know this will not be true, so we negate it
assert not np.equal(A.all(), -1 * A.T.all())