import numpy as np
from numpy.linalg import solve, inv
from scheme_calculator import central, forward_backward

# Stencils must be a even number!
def A_matrix(stencils, derivative, gridpoints):
    s = stencils
    N = gridpoints
    d = derivative

    A = np.zeros((N,N))
    cen = central(s-1,d)[0]
    end = forward_backward(s,d,1)[0]

    l_cen = len(cen)
    l_end = len(end)

    central_endpoints = int((l_cen-1)/2)
    #Inputting forward deriving schemes
    for i in range(central_endpoints):
        A[i, i: i+l_end] = end

    #Inputting backwards deriving schemes
    for i in range(N-central_endpoints, N):
        if derivative % 2 == 0:
            A[i, i - l_end + 1: i + 1 ] = np.flip(end)
        else:
            A[i, i - l_end + 1: i + 1 ] = np.flip(-end)

    #Inputting central deriving schemes
    for i in range(central_endpoints, N-central_endpoints):
        A[i, i-central_endpoints: i+central_endpoints+1] = cen
    
    return A