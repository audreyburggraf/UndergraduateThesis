import numpy as np

from scipy.linalg import lstsq
from scipy.optimize import leastsq
from scipy.optimize import minimize

from functions import *

# -----------------------------------------------------------------------------------------------------------------------------
def signal_func(pars, times):

    # [deg,   deg, mas/year, mas/year, mas, solar masses, unitless, rad, rad, unitless, Jupiter masses, years, years]
    alpha0, delta0, mu_alpha, mu_delta, parallax, m_star, e, omega, Omega, cos_i, m_planet, P_orb, t_peri = pars
   
    # Proper motion signal 
    prop_ra, prop_dec = generate_pm_signal(mu_alpha, mu_delta, times) # [uas]
    
    # Parallax signal 
    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times) # [uas]
    
    # Planet signal
    planetary_pars = parallax, e, omega, Omega, cos_i, m_planet, m_star, P_orb, t_peri # [various]
    planetary_ra, planetary_dec = generate_planet_signal(*planetary_pars, times)        # [uas]
   
    # add ll three to find full signal
    signal_ra  = prop_ra + parallax_ra   + planetary_ra  
    signal_dec = prop_dec + parallax_dec + planetary_dec


    return(signal_ra, signal_dec)
# ------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, ra_obs, dec_obs, times, noise_ra, noise_dec):
    
    ra_pred, dec_pred = signal_func(pars, times)
   
    d_ra  = (ra_obs  - ra_pred) / noise_ra
    d_dec = (dec_obs - dec_pred) / noise_dec

    return np.concatenate((d_ra, d_dec))
# -----------------------------------------------------------------------------------------------------------------------------
def find_chi_squared(y_obs, y_exp, error):
    chi = (y_obs-y_exp)**2/error**2

    chi_squared = np.sum(chi)

    return(chi_squared)
# -----------------------------------------------------------------------------------------------------------------------------



# ------------------------------- N O - P L A N E T - F I T --------------------------------------
def no_planet_fit(pars, signal_ra_obs, signal_dec_obs,noise_ra, noise_dec,N, times): 
    alpha0, delta0, mu_alpha, mu_delta, parallax = pars

    M = np.zeros((2*N, 5))

    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times) # [uas]

    for i in range(N):
        M[i,0] = 1 
        M[i,1] = 0 
        M[i,2] = times[i]
        M[i,3] = 0
        M[i,4] = parallax_ra[i]
        
        M[i+N, 0] = 0
        M[i+N, 1] = 1
        M[i+N, 2] = 0
        M[i+N, 3] = times[i]
        M[i+N, 4] = parallax_dec[i]

    # finding the best fit values for the S samples

    np_best_fit_val_ra, _, _, _ = lstsq(M, signal_ra_obs)
    np_best_fit_val_dec, _, _, _ = lstsq(M, signal_dec_obs)

    # create empty arrays 
    array_ra = np.zeros((N, 5))
    array_dec = np.zeros((N, 5))

    for i in range(N):
        x_ra = np_best_fit_val_ra # x is equal to kth row of np_best_fit_val
        x_dec = np_best_fit_val_dec # x is equal to kth row of np_best_fit_val

        for j in range(5):
            array_ra[i,j] = M[i,j]*x_ra[j]
            array_dec[i,j] = M[i,j]*x_dec[j]

    array_row_sums_ra = np.sum(array_ra, axis=1) 
    array_row_sums_dec = np.sum(array_dec, axis=1)  

    np_chi_sq_ra = np.sum((array_row_sums_ra - signal_ra_obs)**2/noise_ra**2)
    np_chi_sq_dec = np.sum((array_row_sums_dec - signal_dec_obs)**2/noise_dec**2)

    return np_best_fit_val_ra, np_best_fit_val_dec, np_chi_sq_ra, np_chi_sq_dec
    # ----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# ----------------------------- W I T H - P L A N E T - F I T ----------------------------------------------
def one_planet_fit(parameters, signal_ra_obs, signal_dec_obs, noise_ra, noise_dec, times): 
#     alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, m_planet, P_orb, t_peri, m_star = parameters
#     fitting_params = [alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, m_planet, P_orb, t_peri] 
    
    # bounds so the values don't 'run off' 
    bounds = ((0, 360),                       # alpha_0   [degrees]
              (-90,90),                       # delta_0   [degrees] 
              (-np.inf,np.inf),               # mu_alpha  [mas/year]
              (-np.inf,np.inf),               # mu_delta  [mas/year]
              (0, np.inf),                    # parallax  [mas]
              # (0, np.inf),                  # m_star    [Solar masses]
              (0, 1),                         # e         [unitless]
              (0, 2*np.pi),                   # omega     [rad]
              (0, 2*np.pi),                   # Omega     [rad]
              (-1,1),                         # cos_i     [unitless]
              (0,15),                         # LOG SIGNAL AMOLITUDE m_planet  [Jupiter masses]
              (-np.inf, np.inf),              # P_orb     [years]
              (-np.inf, np.inf))              # t_peri    [years]
        
    # getting best/fitted values 
    fitted_params_1P, _, _, _, _ = leastsq(normalized_residuals, parameters, args=(signal_ra_obs, signal_dec_obs, times, noise_ra, noise_dec), full_output=1)

    # if nan values then use other method 
    fn = lambda x: normalized_residuals(x,*args) @ normalized_residuals(x,*args)
    args = (parameters, signal_ra_obs, signal_dec_obs, times, noise_ra, noise_dec)

    if np.isnan(fitted_params_1P).any():
        result_min_P1 = minimize(fn, parameters, method='Nelder-Mead', bounds=bounds)
        fitted_params_1P = result_min_P1.x

    # creating best signal from the best fit parameters 
    signal_ra_best, signal_dec_best = signal_func(fitted_params_1P, times)

    # finding chi squared for with and without planet
    wp_chi_sq_ra = find_chi_squared(signal_ra_best, signal_ra_obs, noise_ra)
    wp_chi_sq_dec = find_chi_squared(signal_dec_best, signal_dec_obs, noise_dec)
    
    return fitted_params_1P, wp_chi_sq_ra, wp_chi_sq_dec
# -----------------------------------------------------------------------------------------------------------------------------        
# ---------------------------- C A L C U L A T I N G - B I C - A N D - D E L T A - B I C---------------------   
def BIC(np_chi_sq, wp_chi_sq, N): 
    np_BIC = np_chi_sq + 5  * np.log(N)
    wp_BIC = wp_chi_sq + 12 * np.log(N)
    
    Delta_BIC = wp_BIC - np_BIC
    
    return Delta_BIC
# ------------------------------------------------------------------------------------------------------------------------------