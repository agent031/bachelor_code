import numpy as np
import astropy.units as u
from astropy.constants import c, k_B, M_sun, G, m_p, sigma_sb, m_p
from scipy.special import iv
from scipy.sparse import csr_matrix
from matrix_calculator import A_matrix

N = 300
r_in = 0.01 # AU
r_out = 1e4 # AU 
r = np.logspace(np.log(r_in), np.log(r_out), N, base = np.exp(1))
r_au = r * u.au
r_log = np.log(r)
Δr_log = r_log[1] - r_log[0]


D1_log = A_matrix(4, 1, N) / Δr_log
sD1_log = csr_matrix(D1_log.copy())


### Keplerian velocity ###
Ω = (np.sqrt((G * M_sun) / r_au**3)).decompose()

### Irradiation temperature T_req ###
T_1au = 280 * u.K
p = -1/2
T_req = T_1au * (r)**p
