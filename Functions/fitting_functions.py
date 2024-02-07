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
              (-2, 1),                        # log_P     [log10(years_]
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
    np_BIC = -2*np_chi_sq + 5  * np.log(N)
    wp_BIC = -2*wp_chi_sq + 12 * np.log(N)
    
    # Calculate Delta BIC 
    Delta_BIC = wp_BIC - np_BIC
    
    # Set the injected and recovered P and m 
    P_rec  = fitted_params_1P[10]
    mp_rec = fitted_params_1P[9]
    
    P_inj  = inj_params_1P[10]
    mp_inj = inj_params_1P[9]
    
    
    # Check to see if the three conditions were met 
    # Step 1: is Delta BIC > 20?
    condition_1 = Delta_BIC <  -20
    
    # Step 2: Is the fitted value of P within some tolerence of the real value? 
    condition_2 = np.isclose(P_inj, P_rec, rtol=0.05) # CHECK LOG PERIOD
    
    # Step 3: Is the fitted value of m_planet within some tolerence of the real value? 
    condition_3 = np.isclose(mp_inj, mp_rec, rtol=0.05)
    
#     detection = ((Delta_BIC > 20) & (np.isclose(P_inj, P_rec, rtol=0.05) & np.isclose(mp_inj, mp_rec, rtol=0.05))).astype(int)

    conditions_satisfied = np.array([condition_1, condition_2, condition_3], dtype=int)
    
    # Combine all conditions into a binary array for detection
    detection = np.all(conditions_satisfied).astype(int)
    
    return (np_BIC, wp_BIC, Delta_BIC, detection, conditions_satisfied) 
# -----------------------------------------------------------------------------------------------------------------------------


def multiple_fitting_function(df, N1, N2, num_of_runs): 
    
    detection_results = []
    
    for i in range(num_of_runs):
    
        # call function
        parameter_result, synthetic_result, model_result, error_result = find_signal_components(df, N1, N2, print_params='neither')

        # break down the result statement 

        # parameter_result includes the no planet and 1 planet parameters in their proper units 
        (inj_params_0P, inj_params_1P) = parameter_result

        # model_results includes the signal components and times for the synthetic in [uas] and [years]
        (prop_ra_synthetic, prop_dec_synthetic, 
         parallax_ra_synthetic, parallax_dec_synthetic, 
         planetary_ra_synthetic, planetary_dec_synthetic, 
         times_synthetic) = synthetic_result

        # model_results includes the signal components and times for the model in [uas] and [years]
        (prop_ra_model, prop_dec_model, 
         parallax_ra_model, parallax_dec_model, 
         planetary_ra_model, planetary_dec_model, 
         times_model) = model_result

        # error_result includes noise in ra and dec direction plus error (same for ra and dec) in [uas]
        (noise_ra, noise_dec, errors) = error_result

        # finding observed signal
        signal_ra_obs  = prop_ra_synthetic + parallax_ra_synthetic + planetary_ra_synthetic + noise_ra
        signal_dec_obs = prop_dec_synthetic + parallax_dec_synthetic + planetary_dec_synthetic + noise_dec


        # zero planet fit 
        fitted_params_0P, np_chi_sq = no_planet_fit(inj_params_0P, signal_ra_obs, signal_dec_obs, noise_ra, noise_dec, times_synthetic)

        # print_parameter_differences(inj_params_0P, fitted_params_0P, return_type='np')

        # one planet fit 
        fitted_params_1P, wp_chi_sq = one_planet_fit(inj_params_1P, signal_ra_obs, signal_dec_obs, noise_ra, noise_dec, times_synthetic)

        # print_parameter_differences(inj_params_1P, fitted_params_1P, return_type='wp')

        # finding detection result 
        detection_result = detection_function(np_chi_sq, wp_chi_sq, inj_params_1P, fitted_params_1P, N1)

        (np_BIC, wp_BIC, Delta_BIC, detection, conditions_satisfied) = detection_result
        
        print(detection_result)
        
        print("run #", i, " conditions_satisfied:", conditions_satisfied)
        
        detection_results.append(detection)
        
        
        
    
    return detection_results
