import numpy as np
import astropy.units as u
import numba

from matrix_calculator import A_matrix
from scheme_calculator import forward_backward, central
from ODE_schemes import stencil_calc


# Numerical calculation of 1. derivative with equal gridspacing
def get_first_dev(array, Δx):
    array_dev = np.zeros_like(array)
    array_dev[0] = forward_backward(3, 1, forward = True)[0] @ array[:3]
    array_dev[-1] = forward_backward(3, 1, forward = False)[0] @ array[-3:]
    array_dev[1: -1] = central(4, 1)[0] @ np.array([array[:-2], array[1: -1], array[2:]])
    return array_dev / Δx


# Numerical calculation of 1. derivative with increasing gridspacing
def get_1_dev_new(array, Δx):
    array_dev = np.zeros_like(array)
    array_dev[0] = forward_backward(3, 1, forward = True)[0] @ (array[:3] / Δx[:3])
    array_dev[-1] = forward_backward(3, 1, forward = False)[0] @ (array[-3:] / Δx[-3:])
    array_dev[1: -1] = central(4, 1)[0] @ np.array([array[:-2], array[1: -1], array[2:]])  
    #array_dev[1: -1] = central(4, 1)[0] @ np.array([array[:-2] / Δx[:-2] , array[1: -1] / Δx[1 : -1], array[2:] / Δx[2:]])  
    return array_dev      


# Initial condition for surface density
def Σ_initial(r, Σ_1au = 1.7e4 * u.g * u.cm**-2, r_cut = 30 * u.au):
    return Σ_1au * (r / (1 * u.au))**(-3/2) * np.exp(- r / r_cut)  


# Defining array with specific gridspacing:
r_in = 0.01 # AU
r_out = 1e4 # AU 
r = r_in
r_list = [r]
while r < r_out:
    Δr = np.sqrt(r)
    r = r + Δr
    r_list.append(r)
r_array = np.asarray(r_list)    


# Making A-matrix with 1. derivative irregular gridpoints 
N = len(r_array)
s = 3

i1 = 0
i2 = s
A = np.zeros((N, N))
for i in range(N):
    if abs(i1 - i) >= s/2 and i2 < N:
        i1 += 1
        i2 += 1

    stencil = r_array[i1:i2] - r_array[i]
    coeff = stencil_calc(stencil, 1)
    A[i, i1:i2] = coeff

first_dev_matrix = A.copy()

def get_1_dev_irr(r):
    return r @ first_dev_matrix

