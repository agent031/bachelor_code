import numpy as np
from numpy.linalg import solve, inv


def central(stencilpoints, derivative):
    start = int((stencilpoints - 1) * 0.5)
    x = np.arange(-start,start + 1)
    x = x[None,:]
    e = np.arange(2*start + 1)[:,None]
    A = x**e
    b = np.zeros(2*start+1)
    b[derivative] = np.math.factorial(derivative)  
    return solve(A,b), stencilpoints+1-derivative


def forward_backward(stencilpoints, derivative, forward=1):
    s = stencilpoints
    x = np.arange(s)
    x = x[None,:]
    e = np.arange(s)[:,None]
    A = x**e
    b = np.zeros(s)
    b[derivative] = np.math.factorial(derivative)
    if forward == 1:
        return solve(A,b), stencilpoints-derivative 
    elif forward == 0:
        return np.flip(solve(A,b)), stencilpoints-derivative


def mixed_stencils(back, forth, derivative):
    b = back
    f = forth
    s = 1 + b + f
    x = np.arange(-b, f + 1)
    x = x[None,:]
    e = np.arange(8)[:,None]
    A = x**e
    b = np.zeros(s)
    b[derivative] = np.math.factorial(derivative)
    return solve(A,b)