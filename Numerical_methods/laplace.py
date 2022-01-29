import numpy as np
import numba

@numba.njit
def laplace_2D(N):
    L = np.zeros((N**2,N**2))
    for i1 in range(N):
        for i2 in range(N):
            for j1 in range(N):
                for j2 in range(N):

                    k = N * i1 + j1
                    l = N * i2 + j2

                    if k == l:
                        L[k,l] = -4
                    if abs(i1 - i2) == 1 and j1 == j2:
                        L[k,l] = 1
                    if abs(j1 - j2) == 1 and i1 == i2:
                        L[k,l] = 1
    return L

@numba.njit
def dim2_array(N):
    x=np.zeros(N**2)
    y=np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            k = N * i + j
            x[k] = j 
            y[k] = i
    return x, y