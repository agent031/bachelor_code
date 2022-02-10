import numpy as np
from numpy.linalg import solve
import numba


def stencil_calc(s, d):
    s_len = len(s)
    s = s[None,:]
    e = np.arange(s_len)[:,None]
    A = s**e
    b = np.zeros(s_len)
    b[d] = np.math.factorial(d)
    return solve(A, b)



def A_matrix_regular(N, s, derivative):
    i1 = 0
    i2 = s
    A = np.zeros((N, N))
    for i in range(N):
        if abs(i1 - i) >= s/2 and i2 < N:
            i1 += 1
            i2 += 1

        stencil = np.arange(i1 - i, i2 - i)
        coeff = stencil_calc(stencil, derivative)
        A[i, i1:i2] = coeff
    return A
        