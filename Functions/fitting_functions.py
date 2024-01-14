import numpy as np

from scipy.linalg import lstsq
from scipy.optimize import leastsq, minimize

from functions import *

# -----------------------------------------------------------------------------------------------------------------------------
def signal_func(pars, times, signal_type):

    # Find the model function in the ra and dec direction based on signal_type
    if signal_type == 'np':
        # [deg,   deg, mas/year, mas/year, mas]
        alpha0, delta0, mu_alpha, mu_delta, parallax = pars
        
    elif signal_type == 'wp':
        # [deg,   deg, mas/year, mas/year, mas, solar masses, unitless, rad, rad, unitless, Jupiter masses, log10(years), years]
        alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, m_planet, log_P, t_peri, m_star = pars
    else:
        raise ValueError("Invalid signal_type. Use 'np' or 'wp'.")
        
        
    # Proper motion signal 
    prop_ra, prop_dec = generate_pm_signal(mu_alpha, mu_delta, times) # [uas]
    
    # Parallax signal 
    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times) # [uas]
    
    
    # Find the model function in the ra and dec direction based on signal_type
    if signal_type == 'np':
        # add all two to find full signal
        signal_ra  = prop_ra  + parallax_ra   
        signal_dec = prop_dec + parallax_dec 
        
    elif signal_type == 'wp':
        # Planet signal
        planetary_pars = parallax, e, omega, Omega, cos_i, m_planet, log_P, t_peri, m_star  # [various]
        planetary_ra, planetary_dec = generate_planet_signal(*planetary_pars, times)        # [uas]
    
        # add all three to find full signal
        signal_ra  = prop_ra  + parallax_ra   + planetary_ra  
        signal_dec = prop_dec + parallax_dec  + planetary_dec
    else:
        raise ValueError("Invalid signal_type. Use 'np' or 'wp'.")
   

    return(signal_ra, signal_dec)
# ------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
def normalized_residuals(pars, signal_synthetic, times, noise, signal_type):
    
    # This function reads in a concatenated array of times, times and we want just a times array 
#     index_of_concatenation = int(len(times_double) / 2)
#     times = times_double[:index_of_concatenation]
    
    # Find the model function in the ra and dec direction based on signal_type
    if signal_type == 'np':
        model_ra, model_dec = signal_func(pars, times, 'np')
        
    elif signal_type == 'wp':
        model_ra, model_dec = signal_func(pars, times, 'wp')
        
    else:
        raise ValueError("Invalid signal_type. Use 'np' or 'wp'.")
    
    # Concatenate it together 
    signal_model = np.concatenate((model_ra, model_dec))
   
    d_signal = (signal_synthetic - signal_model) / noise

    return d_signal
# -----------------------------------------------------------------------------------------------------------------------------
def find_chi_squared(y_synthetic, y_exp, error):
    chi = (y_synthetic-y_exp)**2/error**2

    chi_squared = np.sum(chi)

    return(chi_squared)
# -----------------------------------------------------------------------------------------------------------------------------



# ------------------------------- N O - P L A N E T - F I T --------------------------------------
def no_planet_fit(np_parameters, signal_ra_synthetic, signal_dec_synthetic, noise_ra, noise_dec, times): 
    
    # combining noise in ra and dec directions 
    noise = np.concatenate((noise_ra, noise_dec))
    
    # combining synthetic signal in ra and dec directions 
    signal_synthetic = np.concatenate((signal_ra_synthetic, signal_dec_synthetic))
    
    
    # bounds so the values don't 'run off' 
    bounds = ((0, 360),                       # alpha_0   [degrees]
              (-90,90),                       # delta_0   [degrees] 
              (-np.inf,np.inf),               # mu_alpha  [mas/year]
              (-np.inf,np.inf),               # mu_delta  [mas/year]
              (0, np.inf))                    # parallax  [mas]
        
    # getting best/fitted values 
    fitted_params_0P, _, _, _, _ = leastsq(normalized_residuals, np_parameters, args=(signal_synthetic, times, noise, 'np'), full_output=1)

    # if nan values then use other method 
    fn = lambda x: normalized_residuals(x,*args) @ normalized_residuals(x,*args)
    args = (np_parameters, signal_synthetic, times, noise, 'wp')

    if np.isnan(fitted_params_0P).any():
        result_min_P1 = minimize(fn, parameters, method='Nelder-Mead', bounds=bounds)
        fitted_params_0P = result_min_P1.x

    # creating best signal from the best fit parameters 
    signal_ra_best, signal_dec_best = signal_func(fitted_params_0P, times, 'np')
    signal_best = np.concatenate((signal_ra_best, signal_dec_best))
    
    # finding chi squared for one planet 
    np_chi_sq = find_chi_squared(signal_synthetic, signal_best, noise)
    
    return fitted_params_0P, np_chi_sq
    # ----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# ----------------------------- W I T H - P L A N E T - F I T ----------------------------------------------
def one_planet_fit(parameters, signal_ra_synthetic, signal_dec_synthetic, noise_ra, noise_dec, times): 

    # combining noise in ra and dec directions 
    noise = np.concatenate((noise_ra, noise_dec))
    
    # combining synthetic signal in ra and dec directions 
    signal_synthetic = np.concatenate((signal_ra_synthetic, signal_dec_synthetic))
    
    # making a an array of times, times in years
    # times_double = np.concatenate((times, times))
    
    # bounds so the values don't 'run off' 
    bounds = ((0, 360),                       # alpha_0   [degrees]
              (-90,90),                       # delta_0   [degrees] 
              (-np.inf,np.inf),               # mu_alpha  [mas/year]
              (-np.inf,np.inf),               # mu_delta  [mas/year]
              (0, np.inf),                    # parallax  [mas]
              (0, 1),                         # e         [unitless]
              (0, 2*np.pi),                   # omega     [rad]
              (0, 2*np.pi),                   # Omega     [rad]
              (-1,1),                         # cos_i     [unitless]
              (0,15),                         # LOG SIGNAL AMOLITUDE m_planet  [Jupiter masses]
              (-np.inf, np.inf),              # log_P     [log10(years_]
              (-np.inf, np.inf))              # t_peri    [years]
        
    # getting best/fitted values 
    fitted_params_1P, _, _, _, _ = leastsq(normalized_residuals, parameters, args=(signal_synthetic, times, noise, 'wp'), full_output=1)

    # if nan values then use other method 
    fn = lambda x: normalized_residuals(x,*args) @ normalized_residuals(x,*args)
    args = (parameters, signal_synthetic, times, noise, 'wp')

    if np.isnan(fitted_params_1P).any():
        result_min_P1 = minimize(fn, parameters, method='Nelder-Mead', bounds=bounds)
        fitted_params_1P = result_min_P1.x

    # creating best signal from the best fit parameters 
    signal_ra_best, signal_dec_best = signal_func(fitted_params_1P, times, 'wp')
    signal_best = np.concatenate((signal_ra_best, signal_dec_best))
    
    # finding chi squared for one planet 
    wp_chi_sq = find_chi_squared(signal_synthetic, signal_best, noise)
    
    return fitted_params_1P, wp_chi_sq
# -----------------------------------------------------------------------------------------------------------------------------        
# ---------------------------- C A L C U L A T I N G - B I C - A N D - D E L T A - B I C---------------------   
def detection_function(np_chi_sq, wp_chi_sq, inj_params_1P, fitted_params_1P, N): 
    # the function reads in the chi squared value for the 0 planet and 1 planet fit 
    # the function reads in N, the number of time steps
    
    # Calculate the two BIC values 
    np_BIC = np_chi_sq + 5  * np.log(N)
    wp_BIC = wp_chi_sq + 12 * np.log(N)
    
    # Calculate Delta BIC 
    Delta_BIC = np.abs(np_BIC - wp_BIC)
    
    # Set the injected and recovered P and m 
    P_rec  = fitted_params_1P[10]
    mp_rec = fitted_params_1P[9]
    
    P_inj  = inj_params_1P[10]
    mp_inj = inj_params_1P[9]
    
    
    # Check to see if the three conditions were met 
    # Step 1: is Delta BIC > 20?
    # Step 2: Is the fitted value of P within some tolerence of the real value? 
    # Step 3: Is the fitted value of m_planet within some tolerence of the real value? 
    detection = ((Delta_BIC > 20) & (np.isclose(P_inj, P_rec, rtol=0.05) & np.isclose(mp_inj, mp_rec, rtol=0.05))).astype(int)
    
    return (np_BIC, wp_BIC, Delta_BIC, detection) 
# -----------------------------------------------------------------------------------------------------------------------------