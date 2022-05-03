import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, M_sun, G, m_p, sigma_sb, m_p
from scipy.special import iv
from scipy import stats
from scipy.sparse import csr_matrix
from iminuit import Minuit
import string
import numba
import tqdm

from matrix_calculator import A_matrix
from unchanged_values import r, r_au, sD1_log, Ω, T_req

def analytic_green(x, τ):
    return (np.pi * τ)**(-1) * x**(-1/4) * np.exp(- (1 + x**2) / τ) * iv(1/4, 2*x / τ)
    
# https://docs.scipy.org/doc/scipy/reference/special.html
# Different kinds of bessel functions were tried - "Modified Bessel function of order 1." is the working one with the order sat to 1/4
# (Not including pi in the first part of the soultion gives the same scale on the y-axis as Armitage)


# Initial condition for surface density
def Σ_initial(r, Σ_1au = 1.7e4 * u.g * u.cm**-2, r_cut = 30 * u.au):
    return Σ_1au * (r / (1 * u.au))**(-3/2) * np.exp(- r / r_cut)  


#### Functions to be used many times ####

# The speed of sound
def c_s2(T):
    μ = 2.34
    return ((k_B * T) / (μ * m_p)).decompose()


#### Opacity functions ####
### Opacity ###
#Defining transition region
def kappa_trans(T):
    a, b, c = [ 2.25018776e+00,  3.18833142e-02, -1.47474877e+03]
    return a * (1 - np.tanh((T + c) * b))

### Defining kappa function ###
def kappa_R(T):
    if type(T) != np.ndarray:
        TT = (T.value).copy()
    else:
        TT = T.copy()
    κ = np.zeros_like(TT)
    κ[TT < 150] = 4.5 * (TT[TT < 150] / 150)**2 
    κ[TT >= 150] = kappa_trans(TT[TT >= 150])

    return κ 


def tau_R(T, Σ):
    κ = kappa_R(T).copy()
    return κ * (Σ.to('g/cm2')).value / 2

def tau_P(τ_R):
    τ_P = 2.4 * τ_R.copy()
    τ_P = np.maximum(τ_P, 0.5)
    return τ_P

### Combined Opacity function ###
def opacity(T, Σ):
    return (3/8 * tau_R(T, Σ) + (2 * tau_P(tau_R(T, Σ)))**(-1))**0.25


### F_rad Strond DW
def F_rad_strongDW(T, Σ, α_rφ = 8e-5):
    func_to_der = (r_au**2 * Σ * Ω * α_rφ * c_s2(T)).decompose()
    F_rad = (-(r_au**(-2)) * (sD1_log @ func_to_der) * func_to_der.unit).decompose()
    F_rad_nounit = F_rad.value
    F_rad_nounit[F_rad_nounit <= 0] = 0
    return F_rad_nounit * F_rad.unit


### The viscous temperature ###
def T_vis_strongDW(T, Σ):
    return (opacity(T, Σ) * ((0.5 * sigma_sb**(-1)) * F_rad_strongDW(T, Σ))**(0.25)).decompose()


### Functions for guessing temperatue ###
def guess_T(T, T_vis):
    T_new4 = T_vis(T, Σ_initial(r_au)).value**4 + T_req.value**4
    return T_new4**0.25 * u.K

### Find initial temperature ###
def find_temp(iterations, T0, T_vis):
    T_list = [T0]
    for i in tqdm.tqdm(range(iterations)):
        T_old = T_list[-1]
        T_new = guess_T(T_old, T_vis)
        T_new = np.minimum(T_new, T_old * 1.01)
        T_new = np.maximum(T_new, T_old * 0.99)
        T_damp = (T_old + 2. * T_new)/3.
        T_list.append(T_damp)
    return T_list[-1]


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


### Some AppStat func to write in plots ###
def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None