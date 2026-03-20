import numpy as np
from gram_schmidt import gram_schmidt

if __name__ == '__main__':
    V = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1]
    ], dtype=float)

    Q = gram_schmidt(V)

    # Since Q is orthonormal, Q.T @ Q = I and we can do:
    #     1. V = QR
    #     2. Q.T @ V = Q.T @ Q @ R
    #     3. Q.T @ V = R
    R = Q.T @ V

    assert np.allclose(V, Q @ R)
    print("V = QR")