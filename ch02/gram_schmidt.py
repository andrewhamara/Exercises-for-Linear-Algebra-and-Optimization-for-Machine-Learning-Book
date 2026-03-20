import numpy as np

def gram_schmidt(V):
    row_1_normalized = V[0] / np.sqrt(np.dot(V[0], V[0]))
    Q = [row_1_normalized]

    n_rows = V.shape[0]
    for i in range(1, n_rows):
        Q_i = V[i]

        for k in range(i):
            Vi_projected_onto_Qk = (np.dot(V[i], Q[k]) / np.dot(Q[k], Q[k])) * Q[k]
            Q_i -= Vi_projected_onto_Qk

        Q_i /= np.sqrt(np.dot(Q_i, Q_i))
        Q.append(Q_i)

    return np.array(Q)

if __name__ == '__main__':
    V = np.array([
        [1,1,0],
        [1,0,1],
        [0,1,1]
    ], dtype=float)

    Q = gram_schmidt(V)

    assert np.allclose(Q @ Q.T, np.eye(Q.shape[0]))
    print("Q forms an orthonormal basis for the space spanned by V")