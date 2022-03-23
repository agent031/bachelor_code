import numpy as np
import numba 

@numba.njit
def DOP45(F, x0 , t_start=0, t_final=100, Δt=0.1, ε_rel=1e-3, ε_abs=1e-6):
    x = np.array(x0)
    s = [x]
    Δ = []
    t = t_start
    time = [t]

    while t < t_final: 
        k1 = F(x,t)
        k2 = F(x + k1*Δt/5, t + Δt/5)
        k3 = F(x + 3/40*k1*Δt + 9/40*k2*Δt, t + 3/10*Δt)
        k4 = F(x + 44/45*k1*Δt - 56/15*k2*Δt + 32/9*k3*Δt, t + 4/5*Δt)
        k5 = F(x + 19372/6561*k1*Δt - 25360/2187*k2*Δt + 64448/6561*k3*Δt - 212/729*k4*Δt, t + 8/9*Δt)
        k6 = F(x + 9017/3168*k1*Δt - 355/33*k2*Δt + 46732/5247*k3*Δt + 49/176*k4*Δt - 5103/18656*k5*Δt, t + Δt)
        k7 = F(x + 35/384*k1*Δt - 0*k2*Δt + 500/1113*k3*Δt + 125/192*k4*Δt - 2187/6784*k5*Δt + 11/84*k5*Δt, t + Δt)

        f_4 = x + Δt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        f_5 = x + Δt*(5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7) 

        ε = abs(np.linalg.norm(f_5 - f_4))
        ε_tol = ε_rel*np.min(np.abs(x)) + ε_abs
        

        Δt_new =  0.95*(ε_tol/ε)**0.2 * Δt
        if Δt_new > 4 * Δt:
            Δt_new = 4 * Δt
        if Δt_new < 0.5 * Δt:
            Δt_new = 0.5 * Δt
        if ε < ε_tol:
            s.append(f_5)
            Δ.append(Δt)
            time.append(t)
            x = f_5
            t = t + Δt
        Δt = Δt_new
    return np.array(s), time, Δ
