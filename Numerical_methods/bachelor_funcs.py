import numpy as np
import numba

from matrix_calculator import A_matrix
from scheme_calculator import forward_backward, central


#@numba.njit
def get_first_dev(array, Δx):
    array_dev = np.zeros_like(array)
    array_dev[0] = forward_backward(3, 1, forward = True)[0] @ array[:3]
    array_dev[-1] = forward_backward(3, 1, forward = False)[0] @ array[-3:]
    array_dev[1: -1] = central(4, 1)[0] @ np.array([array[:-2], array[1: -1], array[2:]])
    return array_dev / Δx