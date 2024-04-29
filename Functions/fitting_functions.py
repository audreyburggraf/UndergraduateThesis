import numpy as np

import pandas as pd

from scipy.optimize import minimize

from functions import *
from signal_functions import *

from tqdm import tqdm
import time


# Normalized residuals function for the zero-planet fit 
# ---------------------------------------------------------------------------------------------------------------
def normalized_residuals_0P(parameters_0P, signal_synthetic, times_synthetic, noise):
    """
    Calculate the normalized residuals for the zero-planet fit.

    Parameters:
    parameters_0P (tuple): A tuple containing the parameters needed for the zero-planet fit.
    signal_synthetic (array-like): Array containing synthetic signal values.
    times_synthetic (array-like): Array containing time values.
    noise (array-like): Array containing noise values.
    signal_type (str): Type of signal. Either 'np' for a non-planetary signal or 'wp' for a planetary signal.

    Returns:
    float: The chi-squared value for the zero-planet fit.
    """
        
    # Find the model function in the ra and dec direction based on signal_type
    model_ra, model_dec = signal_func_0P(parameters_0P, times_synthetic)
        
    # Concatenate it together 
    signal_model = np.concatenate((model_ra, model_dec))
   
    d_signal = (signal_synthetic - signal_model) / noise

    return find_chi_squared(signal_synthetic, signal_model, noise)
# ---------------------------------------------------------------------------------------------------------------


# Normalized residuals for the one-planet fit 
# ---------------------------------------------------------------------------------------------------------------
def normalized_residuals_1P(parameters_1P, m_star, signal_synthetic, times_synthetic, noise):
    """
    Calculate the normalized residuals for the one-planet fit.

    Parameters:
    parameters_1P (tuple): A tuple containing the parameters needed for the one-planet fit.
    m_star (float): Mass of the host star in solar masses.
    signal_synthetic (array-like): Array containing synthetic signal values.
    times_synthetic (array-like): Array containing time values.
    noise (array-like): Array containing noise values.
    signal_type (str): Type of signal. Either 'np' for a non-planetary signal or 'wp' for a planetary signal.

    Returns:
    float: The chi-squared value for the one-planet fit.
    """    
    
    # Find the model function in the ra and dec direction based on signal_type
    model_ra, model_dec = signal_func_1P(parameters_1P, m_star, times_synthetic)
    
    # Concatenate it together 
    signal_model = np.concatenate((model_ra, model_dec))
   
    d_signal = (signal_synthetic - signal_model) / noise

    return find_chi_squared(signal_synthetic, signal_model, noise)
# ---------------------------------------------------------------------------------------------------------------


def find_chi_squared(y_synthetic, y_exp, error):
    """
    Calculate the chi-squared value given synthetic and experimental data.

    Parameters:
    y_synthetic (numpy.ndarray): Synthetic data array.
    y_exp (numpy.ndarray): Experimental data array.
    error (numpy.ndarray): Error array.

    Returns:
    float: The chi-squared value.
    """
    chi = (y_synthetic - y_exp)**2 / error**2

    chi_squared = np.sum(chi)

    return chi_squared
# -----------------------------------------------------------------------------------------------------------------------------


# Zero planet fitting function
# ---------------------------------------------------------------------------------------------------------------
def no_planet_fit(inj_params_0P, signal_ra_synthetic, signal_dec_synthetic, noise_ra, noise_dec, times_synthetic): 
    
    # Combine the noise and synthetic signals in the RA and Dec directions into one array each
    noise = np.concatenate((noise_ra, noise_dec))
    signal_synthetic = np.concatenate((signal_ra_synthetic, signal_dec_synthetic))
    
    
    # Make a bounds array to avoid the parameters from 'running off'
    bounds_0P = ((0, 360),                      # alpha_0   [degrees]
                (-90,90),                       # delta_0   [degrees] 
                (-np.inf,np.inf),               # mu_alpha  [mas/year]
                (-np.inf,np.inf),               # mu_delta  [mas/year]
                (0, np.inf))                    # parallax  [mas]
        
    # Arguments for the objective function
    args = (signal_synthetic, times_synthetic, noise)
    
    # Define the objective function for optimization
    fn = lambda x: normalized_residuals_0P(x,*args) #@ normalized_residuals_0P(x,*args)
    
    # Perform optimization using Nelder-Mead method
    result_min_0P = minimize(fn, inj_params_0P, method='Nelder-Mead', bounds=bounds_0P)
    fitted_params_0P = result_min_0P.x

    # Create the best-fit signal from the fitted parameters
    signal_ra_best, signal_dec_best = signal_func_0P(fitted_params_0P, times_synthetic)
    signal_best = np.concatenate((signal_ra_best, signal_dec_best))
    
    # Calculate the chi-squared value for the one-planet fit
    np_chi_sq = find_chi_squared(signal_synthetic, signal_best, noise)
    
    return fitted_params_0P, np_chi_sq
# ---------------------------------------------------------------------------------------------------------------

# One planet fitting function
# ---------------------------------------------------------------------------------------------------------------
def one_planet_fit(inj_params_1P, m_star, signal_ra_synthetic, signal_dec_synthetic, noise_ra, noise_dec, times_synthetic): 
    
    # Combine the noise and synthetic signals in the RA and Dec directions into one array each
    noise = np.concatenate((noise_ra, noise_dec))
    signal_synthetic = np.concatenate((signal_ra_synthetic, signal_dec_synthetic))
    
    
    # Make a bounds array to avoid the parameters from 'running off'
    bounds_1P = ((0, 360),                      # alpha_0       [degrees]
                (-90,90),                       # delta_0       [degrees] 
                (-np.inf,np.inf),               # mu_alpha      [mas/year]
                (-np.inf,np.inf),               # mu_delta      [mas/year]
                (0, np.inf),                    # parallax      [mas]
                (0, 1),                         # e             [unitless]
                (0, 2*np.pi),                   # omega         [rad]
                (0, 2*np.pi),                   # Omega         [rad]
                (0, 1),                         # cos_i         [unitless]
                (-2,np.log10(15)),              # log_m_planet  [log10(Jupiter masses)] (bounds: 0.01, 15 M_J)
                (-2, 1),                        # log_P         [log10(years)] (bounds: 0.01 to 10 years)
                (-np.inf, np.inf))              # t_peri        [years] 

    # Arguments for the objective function
    args = (m_star, signal_synthetic, times_synthetic, noise)
    
    # Define the objective function for optimization
    fn = lambda x: normalized_residuals_1P(x,*args) #@ normalized_residuals_1P(x,*args)

    # pdb.set_trace()
   
    
    # Perform optimization using Nelder-Mead method
    result_min_1P = minimize(fn, inj_params_1P, method='Nelder-Mead', bounds=bounds_1P)
    fitted_params_1P = result_min_1P.x
    
    # Add a condition to keep track of success of the fit 
    success_1P_fit = int(result_min_1P.success)
    
  
    
    #pdb.set_trace()

    # Create the best-fit signal from the fitted parameters
    signal_ra_best, signal_dec_best = signal_func_1P(fitted_params_1P, m_star, times_synthetic)
    signal_best = np.concatenate((signal_ra_best, signal_dec_best))
    
    
    # Calculate the chi-squared value for the one-planet fit
    wp_chi_sq = find_chi_squared(signal_synthetic, signal_best, noise)
    
    return fitted_params_1P, wp_chi_sq, success_1P_fit
# ---------------------------------------------------------------------------------------------------------------


# Detection function
# ---------------------------------------------------------------------------------------------------------------
def detection_function(np_chi_sq, wp_chi_sq, inj_params_1P, fitted_params_1P, success_1P_fit, N): 
    """
    Determine the detection of a planet based on Bayesian Information Criterion (BIC) comparison and parameter fitting.

    Parameters:
    np_chi_sq (float): Chi-squared value for the model without a planet.
    wp_chi_sq (float): Chi-squared value for the model with a planet.
    inj_params_1P (tuple): Injected parameters for the model with a planet.
    fitted_params_1P (tuple): Fitted parameters for the model with a planet.
    N (int): Number of time steps.

    Returns:
    tuple: A tuple containing the following:
        - np_BIC (float): BIC value for the model without a planet.
        - wp_BIC (float): BIC value for the model with a planet.
        - Delta_BIC (float): Difference in BIC values between the two models.
        - detection (int): Binary indicator (1 if detection criteria met, 0 otherwise).
        - conditions_satisfied (numpy.ndarray): Binary array indicating if each condition for detection is satisfied.
    """
        
    # Calculate the two BIC values 
    BIC_0P = np_chi_sq + 5  * np.log(N) # gaussian equation -2*(-1/2 chi^2)
    BIC_1P = wp_chi_sq + 12 * np.log(N)
    
    # Calculate Delta BIC 
    Delta_BIC = BIC_1P - BIC_0P
    
    # Check to see if the four conditions were met 
    # ---------------------------------------------------------------------------  
    # Condition 1 : ΔBIC < -20
    condition_1 = Delta_BIC < -20
    
    # Condition 2: Recovered P within 5% error of injected P
    condition_2 = np.isclose(inj_params_1P[10][0], fitted_params_1P[10], rtol=0.05) 
    
    # Step 3: Recovered m_p within 5% error of injected m_p
    condition_3 = np.isclose(inj_params_1P[9][0], fitted_params_1P[9], rtol=0.05)
    
    # Step 4: Did minimize do the fit successfully
    condition_4 = (success_1P_fit == 1)
    
    conditions_satisfied = np.array([condition_1, condition_2, condition_3, condition_4], dtype=int)
    
    # pdb.set_trace()

    # Combine all conditions into a binary array for detection
    detection = np.all(conditions_satisfied).astype(int)
    # ---------------------------------------------------------------------------  

    
    return (BIC_0P, BIC_1P, Delta_BIC, detection, conditions_satisfied)
# ---------------------------------------------------------------------------------------------------------------

def multiple_fitting_function(df, N_synthetic, N_model, num_of_runs, filename, save_to_file = False, print_detection_results = False): 
    """
    Perform multiple fitting iterations and determine the detection of a planet for each iteration.

    Parameters:
    df (DataFrame): Input DataFrame containing data for fitting.
    N_synthetic (int): Number of time steps for synthetic data.
    N_model (int): Number of time steps for model data.
    num_of_runs (int): Number of fitting iterations to perform.

    Returns:
    list: A list containing detection results for each iteration.
    """
    
    # Create an empty list to store data from each iteration
    data_list = []
    
    for i in tqdm(range(num_of_runs)):
    
        # Call the signal that finds parameters and makes the signal components and unpack its result statement 
        # ----------------------------------------------------------------------------------------
        inj_params_0P, inj_params_1P, synthetic_signal, model_signal, error_components, alpha, m_star = \
            find_signal_components(df, N_synthetic, N_model, print_params=False, print_alpha=False)


        # Unpack the result statement
        # ----------------------------------------------------------------------------------------
        # Unpack synthetic signal components and times
        (prop_ra_synthetic, prop_dec_synthetic, 
         parallax_ra_synthetic, parallax_dec_synthetic, 
         planetary_ra_synthetic, planetary_dec_synthetic, 
         times_synthetic) = synthetic_signal

        # Unpack model signal components and times
        (prop_ra_model, prop_dec_model, 
         parallax_ra_model, parallax_dec_model, 
         planetary_ra_model, planetary_dec_model, 
         times_model) = model_signal

        # Unpack noise and error components
        (noise_ra, noise_dec, errors) = error_components
        # ----------------------------------------------------------------------------------------
        
        # Find signal to noise - needs organization
        # ----------------------------------------------------------------------------------------
        # signal is astrometric_signal calculated with the injected parameters, or alpha 
        
        
        # THIS IS WHAT I WAS DOING FOR NOISE BEFORE 
        # ----------------------------------------------------------------------------------------
        noise = errors[0] * np.sqrt(2) # quadature
        # ----------------------------------------------------------------------------------------
  
        # it is scaled by a factor 
        # 10**inj_params_1P = inj P [years]
        # times_model.max() should = 5
        scaling_factor = np.sqrt(times_model.max()/10**inj_params_1P[10]) 
        
        # find S/N
        SN = alpha/noise*scaling_factor # unitless
        
        # ----------------------------------------------------------------------------------------
        
        # Calculate the observed function from the synthetic data and noise 
        # ----------------------------------------------------------------------------------------
        signal_ra_obs  = prop_ra_synthetic  + parallax_ra_synthetic  + planetary_ra_synthetic  + noise_ra  # [uas]
        signal_dec_obs = prop_dec_synthetic + parallax_dec_synthetic + planetary_dec_synthetic + noise_dec # [uas]
        # ----------------------------------------------------------------------------------------
        
        # Run the fitting functions 
        # ----------------------------------------------------------------------------------------
        # Zero planet fit 
        fitted_params_0P, np_chi_sq = no_planet_fit(
            inj_params_0P,
            signal_ra_obs, signal_dec_obs,
            noise_ra, noise_dec,
            times_synthetic)

        # One planet fit 
        fitted_params_1P, wp_chi_sq, success_1P_fit = one_planet_fit(
            inj_params_1P,
            m_star,
            signal_ra_obs, signal_dec_obs,
            noise_ra, noise_dec,
            times_synthetic)
        # ----------------------------------------------------------------------------------------
        
        
        # Finding detection result 
        # ----------------------------------------------------------------------------------------
        detection_result = detection_function(np_chi_sq, wp_chi_sq, inj_params_1P, fitted_params_1P, success_1P_fit, N_synthetic)

        (np_BIC, wp_BIC, Delta_BIC, detection, conditions_satisfied) = detection_result
        
        # Print the detection results and conditions satiftied if print_detection_results is True      
        if print_detection_results:
            print('Run #', i, ' Detection:', detection, 'Conditions satisfied:', conditions_satisfied)
             
        # ----------------------------------------------------------------------------------------
        
        # Append the data from each iteration to the data_list
        # ----------------------------------------------------------------------------------------
        data_list.append({
            # Injected 0P 
            'Inj.0P alpha0': float(inj_params_0P[0]),  # [deg]
            'Inj.0P delta0': float(inj_params_0P[1]),  # [deg]
            'Inj.0P pmra'  : float(inj_params_0P[2]),  # [mas/year]
            'Inj.0P pmdec' : float(inj_params_0P[3]),  # [mas/year]
            'Inj.0P prlx'  : float(inj_params_0P[4]),  # [mas]
            # Injected 1P 
            'Inj.1P alpha0'  : float(inj_params_1P[0]),  # [deg]
            'Inj.1P delta0'  : float(inj_params_1P[1]),  # [deg]
            'Inj.1P pmra'    : float(inj_params_1P[2]),  # [mas/year]
            'Inj.1P pmdec'   : float(inj_params_1P[3]),  # [mas/year]
            'Inj.1P prlx'    : float(inj_params_1P[4]),  # [mas]
            'Inj.1P e'       : float(inj_params_1P[5]),  # [unitless]
            'Inj.1P omega'   : float(inj_params_1P[6]),  # [rad]
            'Inj.1P Omega'   : float(inj_params_1P[7]),  # [rad]
            'Inj.1P cosi'    : float(inj_params_1P[8]),  # [unitless]
            'Inj.1P log(m_p)': float(inj_params_1P[9]),  # [log10(MJ)]
            'Inj.1P log(P)'  : float(inj_params_1P[10]), # [log10(years)]
            'Inj.1P tp'      : float(inj_params_1P[11]), # [years]
            # Recovered 0P 
            'Rec.0P alpha0': fitted_params_0P[0], # [deg]
            'Rec.0P delta0': fitted_params_0P[1], # [deg]
            'Rec.0P pmra'  : fitted_params_0P[2], # [mas/year]
            'Rec.0P pmdec' : fitted_params_0P[3], # [mas/year]
            'Rec0P prlx'   : fitted_params_0P[4], # [mas]
            # Re.covered 1P
            'Rec.1P alpha0'  : fitted_params_1P[0],  # [deg]
            'Rec.1P delta0'  : fitted_params_1P[1],  # [deg]
            'Rec.1P pmra'    : fitted_params_1P[2],  # [mas/year]
            'Rec.1P pmdec'   : fitted_params_1P[3],  # [mas/year]
            'Rec.1P prlx'    : fitted_params_1P[4],  # [mas]
            'Rec.1P e'       : fitted_params_1P[5],  # [unitless]
            'Rec.1P omega'   : fitted_params_1P[6],  # [rad]
            'Rec.1P Omega'   : fitted_params_1P[7],  # [rad]
            'Rec.1P cosi'    : fitted_params_1P[8],  # [unitless]
            'Rec.1P log(m_p)': fitted_params_1P[9],  # [log10(MJ)]
            'Rec.1P log(P)'  : fitted_params_1P[10], # [log10(years)]
            'Rec.1P tp'      : fitted_params_1P[11], # [years]
            # Other
            'np_chi_sq': np_chi_sq,
            'wp_chi_sq': wp_chi_sq,
            # BIC values 
            'npBIC': np_BIC,
            'wpBIC': wp_BIC,
            'DeltaBIC': Delta_BIC,
            # Detection 
            'Condition 1': conditions_satisfied[0],
            'Condition 2': conditions_satisfied[1],
            'Condition 3': conditions_satisfied[2],
            'Condition 4': conditions_satisfied[3],
            'Detection' : detection,
            # astrometric signature 
            'Astrometric Signature': float(alpha), # [uas]
            # star mass 
            'Stellar Mass': float(m_star), # [M_solar]
            # Scaling Factor
            'Scaling Factor': float(scaling_factor), # unitless
            # Distance
            'Distance': float(calculate_distance(inj_params_1P[4])), # [pc]
            # Semi-Major Axis 
            'Semi-Major Axis': float(calculate_semi_major_axis(inj_params_1P[10], m_star)), # [AU]
            # Sigma fov
            'Sigma fov': float(errors[0]), # []
            # Noise
            'Noise': float(noise), # [uas]
            # Signal/Noise
            'S/N': float(SN) # [unitless]
        })
        # ----------------------------------------------------------------------------------------
       
        
    # Create a DataFrame from the data_list
    df = pd.DataFrame(data_list)
    
    # file path 
    folder_path = '/Users/audreyburggraf/Desktop/THESIS/Data Files/'
    file_path = folder_path + filename
    
    # Save the DataFrame to a CSV file if save_to_file is True
    if save_to_file:
        df.to_csv(file_path, index=False)
    
        
        
# ---------------------------------------------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------------------------------------------
def detection_function(np_chi_sq, wp_chi_sq, inj_params_1P, fitted_params_1P, success_1P_fit, N): 
    """
    Determine the detection of a planet based on Bayesian Information Criterion (BIC) comparison and parameter fitting.

    Parameters:
    np_chi_sq (float): Chi-squared value for the model without a planet.
    wp_chi_sq (float): Chi-squared value for the model with a planet.
    inj_params_1P (tuple): Injected parameters for the model with a planet.
    fitted_params_1P (tuple): Fitted parameters for the model with a planet.
    N (int): Number of time steps.

    Returns:
    tuple: A tuple containing the following:
        - np_BIC (float): BIC value for the model without a planet.
        - wp_BIC (float): BIC value for the model with a planet.
        - Delta_BIC (float): Difference in BIC values between the two models.
        - detection (int): Binary indicator (1 if detection criteria met, 0 otherwise).
        - conditions_satisfied (numpy.ndarray): Binary array indicating if each condition for detection is satisfied.
    """
        
    # Calculate the two BIC values 
    BIC_0P = np_chi_sq + 5  * np.log(N) # gaussian equation -2*(-1/2 chi^2)
    BIC_1P = wp_chi_sq + 12 * np.log(N)
    
    # Calculate Delta BIC 
    Delta_BIC = BIC_1P - BIC_0P
    
    # Check to see if the four conditions were met 
    # ---------------------------------------------------------------------------  
    # Condition 1 : ΔBIC < -20
    condition_1 = Delta_BIC < -20
    
    # Condition 2: Recovered P within 5% error of injected P
    condition_2 = np.isclose(inj_params_1P[10][0], fitted_params_1P[10], rtol=0.05) 
    
    # Step 3: Recovered m_p within 5% error of injected m_p
    condition_3 = np.isclose(inj_params_1P[9][0], fitted_params_1P[9], rtol=0.05)
    
    # Step 4: Did minimize do the fit successfully
    condition_4 = (success_1P_fit == 1)
    
    conditions_satisfied = np.array([condition_1, condition_2, condition_3, condition_4], dtype=int)
    
    # pdb.set_trace()

    # Combine all conditions into a binary array for detection
    detection = np.all(conditions_satisfied).astype(int)
    # ---------------------------------------------------------------------------  

    
    return (BIC_0P, BIC_1P, Delta_BIC, detection, conditions_satisfied)
# ---------------------------------------------------------------------------------------------------------------

def HARDCODED_multiple_fitting_function(m_planet_HARDCODED, P_HARDCODED,m_star_HARDCODED, df, N_synthetic, N_model, num_of_runs, filename, save_to_file = False, print_detection_results = False): 
    """
    Perform multiple fitting iterations and determine the detection of a planet for each iteration.

    Parameters:
    df (DataFrame): Input DataFrame containing data for fitting.
    N_synthetic (int): Number of time steps for synthetic data.
    N_model (int): Number of time steps for model data.
    num_of_runs (int): Number of fitting iterations to perform.

    Returns:
    list: A list containing detection results for each iteration.
    """
    
    # Create an empty list to store data from each iteration
    data_list = []
    
    for i in tqdm(range(num_of_runs)):
    
        # Call the signal that finds parameters and makes the signal components and unpack its result statement 
        # ----------------------------------------------------------------------------------------
        inj_params_0P, inj_params_1P, synthetic_signal, model_signal, error_components, alpha, m_star = \
            HARDCODED_find_signal_components(m_planet_HARDCODED, P_HARDCODED,m_star_HARDCODED, df, N_synthetic, N_model, print_params=True, print_alpha=True)


        # Unpack the result statement
        # ----------------------------------------------------------------------------------------
        # Unpack synthetic signal components and times
        (prop_ra_synthetic, prop_dec_synthetic, 
         parallax_ra_synthetic, parallax_dec_synthetic, 
         planetary_ra_synthetic, planetary_dec_synthetic, 
         times_synthetic) = synthetic_signal

        # Unpack model signal components and times
        (prop_ra_model, prop_dec_model, 
         parallax_ra_model, parallax_dec_model, 
         planetary_ra_model, planetary_dec_model, 
         times_model) = model_signal

        # Unpack noise and error components
        (noise_ra, noise_dec, errors) = error_components
        # ----------------------------------------------------------------------------------------
        
        # Find signal to noise - needs organization
        # ----------------------------------------------------------------------------------------
        # signal is astrometric_signal calculated with the injected parameters, or alpha 
        
        
        # THIS IS WHAT I WAS DOING FOR NOISE BEFORE 
        # ----------------------------------------------------------------------------------------
        noise = errors[0] * np.sqrt(2) # quadature
        # ----------------------------------------------------------------------------------------
  
        # it is scaled by a factor 
        # 10**inj_params_1P = inj P [years]
        # times_model.max() should = 5
        scaling_factor = np.sqrt(times_model.max()/10**inj_params_1P[10]) 
        
        # find S/N
        SN = alpha/noise*scaling_factor # unitless
        
        # ----------------------------------------------------------------------------------------
        
        # Calculate the observed function from the synthetic data and noise 
        # ----------------------------------------------------------------------------------------
        signal_ra_obs  = prop_ra_synthetic  + parallax_ra_synthetic  + planetary_ra_synthetic  + noise_ra  # [uas]
        signal_dec_obs = prop_dec_synthetic + parallax_dec_synthetic + planetary_dec_synthetic + noise_dec # [uas]
        # ----------------------------------------------------------------------------------------
        
        # Run the fitting functions 
        # ----------------------------------------------------------------------------------------
        # Zero planet fit 
        fitted_params_0P, np_chi_sq = no_planet_fit(
            inj_params_0P,
            signal_ra_obs, signal_dec_obs,
            noise_ra, noise_dec,
            times_synthetic)

        # One planet fit 
        fitted_params_1P, wp_chi_sq, success_1P_fit = one_planet_fit(
            inj_params_1P,
            m_star,
            signal_ra_obs, signal_dec_obs,
            noise_ra, noise_dec,
            times_synthetic)
        # ----------------------------------------------------------------------------------------
        
        
        # Finding detection result 
        # ----------------------------------------------------------------------------------------
        detection_result = detection_function(np_chi_sq, wp_chi_sq, inj_params_1P, fitted_params_1P, success_1P_fit, N_synthetic)

        (np_BIC, wp_BIC, Delta_BIC, detection, conditions_satisfied) = detection_result
        
        # Print the detection results and conditions satiftied if print_detection_results is True      
        if print_detection_results:
            print('Run #', i, ' Detection:', detection, 'Conditions satisfied:', conditions_satisfied)
             
        
        
        
        
