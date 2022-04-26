import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, M_sun, G, m_p, sigma_sb, m_p
from scipy.special import iv
from scipy import stats
from scipy.sparse import csr_matrix
from iminuit import Minuit
import string
import numba

from matrix_calculator import A_matrix
from scheme_calculator import forward_backward, central
from ODE_schemes import stencil_calc

def analytic_green(x, τ):
    return (np.pi * τ)**(-1) * x**(-1/4) * np.exp(- (1 + x**2) / τ) * iv(1/4, 2*x / τ)
    
# https://docs.scipy.org/doc/scipy/reference/special.html
# Different kinds of bessel functions were tried - "Modified Bessel function of order 1." is the working one with the order sat to 1/4
# (Not including pi in the first part of the soultion gives the same scale on the y-axis as Armitage)



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


#### Presets for plotting ####
color_use = ['orangered', 'cornflowerblue', 'tab:orange', 'seagreen', 'blueviolet', 'olive']

#### Functions to be used many times ####

# The speed of sound
def c_s2(T):
    μ = 2.34
    return ((k_B * T) / (μ * m_p)).decompose()



#### Opacity functions ####
### Opacity ###
def tau_R(T, Σ):
    if type(T) != np.ndarray:
        TT = (T.value).copy()
    else:
        TT = T.copy()
    κ = np.zeros_like(TT)

    κ[TT < 150] = 4.5 * (TT[TT < 150] / 150)**2 

    κ[np.where((TT >= 150) & (TT <= 1500))] = 4.5 

    #Defining transition region
    def τ_R_func(T):
        a, b, c = [ 2.25221001e+00,  1.39416392e-02, -1.72380082e+03]
        return a * (1 - np.tanh((T + c) * b))
    
    Trans_i = np.where((TT >= 1500) & (TT <= 2000))
    κ[Trans_i] = τ_R_func(TT[Trans_i])
    
    κ[TT > 2000] = 0 # Støvet fordamper og går i stykker 

    return κ * (Σ.to('g/cm2')).value / 2


def tau_P(τ_R):
    τ_P = τ_R.copy()
    τ_P[2.4 * τ_P <= 0.5] = 0.5
    τ_P[τ_P != 0.5] = 2.4 * τ_P[τ_P != 0.5]
    return τ_P



#### Calculate Chi2 with "normal" function ####
# Returns: fit, values, errors, X, Y
# X and Y are arrays with the x-values and function-values

def chi2_fit(func, x, y, sigma_y, initial_geuss):
    
    N_var = len(initial_geuss)
    var_name = list(string.ascii_lowercase)
    var = np.array([var_name[i] for i in range(N_var)])

    def chi2_owncalc(*var):
        y_fit = func(x, *var)
        chi2 = np.sum(((y - y_fit) / sigma_y)**2)
        return chi2

    minuit = Minuit(chi2_owncalc, *initial_geuss)
    minuit.errordef = 1.0   
    minuit.migrad();

    Nvar = minuit.nfit                     
    Ndof_fit = len(x) - Nvar 

    Chi2_fit = minuit.fval                          
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)

    X = np.linspace(np.min(x), np.max(x), 1000)
    Y = func(X, *minuit.values[:])
    if type(Y) == float:
        Y = Y * np.ones_like(X)

    return [np.array([Chi2_fit, Ndof_fit, Prob_fit]),
            np.array(minuit.values[:]),
            np.array(minuit.errors[:]),
            np.asarray(minuit.covariance),  
            X, Y ]