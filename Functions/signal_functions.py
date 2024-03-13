import numpy as np
from numpy import cos, sin

import pandas

import warnings, pdb
warnings.filterwarnings('error')

from functions import *
from unit_conversion_functions import *
from functions import calculate_distance

# signal functions 
# -----------------------------------------------------------------------------------------------------------------------------
def generate_pm_signal(mu_alpha, mu_delta, t):
    """
    Generate proper motion signals for right ascension and declination.

    This function calculates the proper motion signals for a celestial object in both
    right ascension (RA) and declination (DEC) based on the input parameters.

    Parameters:
    mu_alpha (float): The proper motion in RA in milliarcseconds per year (mas/year).
    mu_delta (float): The proper motion in DEC in milliarcseconds per year (mas/year).
    t (array-like): An array of time values representing the time elapsed since the initial epoch in years.

    Returns:
    tuple: A tuple containing two elements:
        - pm_term_ra (float): The proper motion signal in RA in microarcseconds (uas).
        - pm_term_dec (float): The proper motion signal in DEC in microarcseconds (uas).
    """   
    
    # Calculate proper motion in radians
    pm_term_ra  = mu_alpha * t                                     # [mas] = [mas/year] * [year]
    pm_term_dec = mu_delta * t                                     # [mas] = [mas/year] * [year]

    # Convert units from radians to microarcseconds (uas)   
    pm_term_ra = milliarcseconds_to_microarcseconds(pm_term_ra)    # [uas] from [mas]
    pm_term_dec = milliarcseconds_to_microarcseconds(pm_term_dec)  # [uas] from [mas]

    return pm_term_ra, pm_term_dec                                 # [uas]
# -----------------------------------------------------------------------------------------------------------------------------
def generate_parallax_signal(alpha0, delta0, parallax, times, a_earth =1):
    """
    Calculate the components of the parallax signal in microarcseconds (uas).

    Parameters:
    - alpha0 (float): Right Ascension (in degrees).
    - delta0 (float): Declination (in degrees).
    - parallax (float): Parallax angle (in milliarcseconds converted to radians).
    - times (float): Time in years.
    - a_earth (float, optional): Distance from the Earth to the Sun in astronomical units (AU). Default is 1 AU.

    Returns:
    - prlx_term_ra (float): Right Ascension component of the parallax signal in microarcseconds (uas).
    - prlx_term_dec (float): Declination component of the parallax signal in microarcseconds (uas).
    """
    
    # Change units 
    alpha0 = degrees_to_radians(alpha0)             # [radians] from [degrees]
    delta0 = degrees_to_radians(delta0)             # [radians] from [degrees]
    parallax = milliarcseconds_to_arcseconds(parallax) # [arcseconds] from [mas]
    
    # Calculate distance factor
    #UNITS???? 1 pc = 1 AU/ 1 arcsec
    d = a_earth/parallax # [AU/arcsec] = [pc]

    # Trigonometric functions
    sind, cosd, sina, cosa  = sin(delta0), cos(delta0), sin(alpha0), cos(alpha0) # [unitless]

    # time 
    T = times # [years]

    # Calculate components of the parallax signal
    prlx_term_ra  = (a_earth/(d*cosd))*(sina*cos(2*np.pi*T)-cosa*sin(2*np.pi*T))   # [arcseconds]
    prlx_term_dec = (a_earth/d)*sind*cos(2*np.pi*T-alpha0)                         # [arcseconds]
       
    
    # Change units to microarcseocnds 
    prlx_term_ra  = arcseconds_to_microarcseconds(prlx_term_ra)  # [uas] from [arcseconds]
    prlx_term_dec = arcseconds_to_microarcseconds(prlx_term_dec) # [uas] from [arcseconds]

    return prlx_term_ra, prlx_term_dec
# -----------------------------------------------------------------------------------------------------------------------------
def calculate_thiele_innes(omega, Omega, cos_i, parallax, log_m_planet, m_star, log_P):
    """
    Calculate the Thiele-Innes constants for a binary star system with an orbiting planet.

    This function takes orbital parameters and masses of the star and planet as input and
    computes the Thiele-Innes constants (A, B, F, G, H, C, and semi-major axis) for the system.

    Parameters:
    omega (float): Argument of periastron in radians.
    Omega (float): Longitude of ascending node in radians.
    cos_i (float): Cosine of the orbital inclination angle (unitless).
    parallax (float): Parallax of the system in milliarcseconds (mas).
    log_m_planet (float): Log10 of the mass of the planet in log Jupiter masses (log10(Mjupiter)).
    m_star (float): Mass of the star in solar masses (Msun).
    log_P (float): Orbital period of the planet in log10(years).

    Returns:
    tuple: A tuple containing Thiele-Innes constants:
        - B (float): Thiele-Innes constant [mas].
        - A (float): Thiele-Innes constant [mas].
        - F (float): Thiele-Innes constant [mas].
        - G (float): Thiele-Innes constant [mas].
        - H (float): Thiele-Innes constant [mas].
        - C (float): Thiele-Innes constant [mas].
    """
    
    # Find distance in pc from parallax in mas
    d = calculate_distance(parallax) # [pc]
    
        
    # Find alpha in mas from m_planet [M_Jup], m_star [M_sun], P [years], d [pc]
    alpha = calculate_astrometric_signature(log_m_planet, m_star, log_P, d) # [mas]
    
    # Find sin(inclination)
#     try:
#         sin_i = np.sqrt(1 - cos_i**2)  # [unitless]
#     except RuntimeWarning:
#         pdb.set_trace()
        
    sin_i = np.sqrt(1 - cos_i**2)  # [unitless]   
    
    A = alpha * (cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cos_i)   # [uas]
    F = alpha * (-sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i)  # [uas]

    B = alpha * (cos(omega) * sin(Omega) + sin(omega) * cos(Omega) * cos_i)   # [uas]
    G = alpha * (-sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i)  # [uas]

    C = alpha * sin(omega) * sin_i  # [uas]
    H = alpha * cos(omega) * sin_i  # [uas]
    
    
    return B, A, F, G, H, C
# -----------------------------------------------------------------------------------------------------------------------------
def generate_planet_signal(planetary_parameters, parallax, m_star, times):
    """
    Generates the astrometric signal of a planet orbiting a star.

    Parameters:
    - parallax_mas (float): Parallax in milliarcseconds (mas).
    - e (float): Orbital eccentricity (unitless).
    - omega (float): Argument of periastron in radians.
    - Omega (float): Longitude of the ascending node in radians.
    - cos_i (float): Cosine of the inclination angle (unitless).
    - log_m_planet (float): Log10 of the mass of the planet in Jupiter masses (log(Mjupiter)).
    - m_star (float): Mass of the star in Solar masses (Msun).
    - log_P (float): Orbital period of the planet in log10(years).
    - t_peri (float): Time of periastron passage in years.
    - times (array-like): Array of time values at which to compute the signal in years.

    Returns:
    - plnt_term_ra (array): Astrometric signal in right ascension (microarcseconds).
    - plnt_term_dec (array): Astrometric signal in declination (microarcseconds).
    """
    
    e, omega, Omega, cos_i, log_m_planet, log_P, t_peri = planetary_parameters 
    
    # Thiele-Innes constants 
    B, A, F, G, _, _ = calculate_thiele_innes(omega, Omega, cos_i, parallax, log_m_planet, m_star, log_P) # [uas]
    
    # find P_orb in years from log10(P) in log10(years)
    P_orb = 10**log_P                           # [years]
        
    # Anomalies?
    try:
        M = (2*np.pi)*(times - t_peri)/P_orb   # [radians]
    except RuntimeWarning:
        pdb.set_trace()


    E = np.vectorize(rebound.M_to_E)(e,M)  # [radians]

    X = (cos(E)-e)               # [unitless]
    
    Yissues = []

    # Check if square root operation will encounter invalid values
    if np.isnan(1 - e ** 2) or np.isinf(1 - e ** 2) or (1 - e ** 2) < 0:
        Yissues.append('Invalid value encountered in sqrt')

    if Yissues:
        Yissues_string = ', '.join(Yissues)
        print("Issues encountered:", Yissues_string)
        Y = 1
    else:
        Y = np.sqrt(1 - e ** 2) * np.sin(E)  # [unitless]
     
    plnt_term_ra  = B*X + G*Y  # [mas] * [unitless] = [uas]
    plnt_term_dec = A*X + F*Y  # [mas] * [unitless] = [uas]

    return plnt_term_ra, plnt_term_dec
# ----------------------------------------------------------------------------------------------------------------------------



# Function to calculate the signal for a zero-planet system based on the provided parameters and time values
# ---------------------------------------------------------------------------------------------------------------
def signal_func_0P(parameters_0P, times):
    """
    Calculate the signal function for a non-planetary signal based on the parameters.

    Parameters:
    pars (tuple): A tuple containing the parameters needed for the signal calculation.
        - alpha0 (float): Right ascension of the source in degrees.
        - delta0 (float): Declination of the source in degrees.
        - mu_alpha (float): Proper motion in right ascension in milliarcseconds per year.
        - mu_delta (float): Proper motion in declination in milliarcseconds per year.
        - parallax (float): Parallax angle in milliarcseconds.
    times (array-like): Array of time values.

    Returns:
    tuple: A tuple containing the signal in right ascension (signal_ra) and declination (signal_dec) directions.
    """
    # Unpack the parameters_0P array
    alpha0, delta0, mu_alpha, mu_delta, parallax = parameters_0P
    
    # Calculate the proper motion term 
    prop_ra, prop_dec = generate_pm_signal(mu_alpha, mu_delta, times)  # [uas]
    
    # Calculate the parallax term
    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)  # [uas]
    
    # Add the two terms together 
    signal_ra = prop_ra + parallax_ra    # [uas]
    signal_dec = prop_dec + parallax_dec # [uas]
    
    return signal_ra, signal_dec # [uas, uas]
# ---------------------------------------------------------------------------------------------------------------


# Function to calculate the signal for a one-planet system based on the provided parameters and time values
# ---------------------------------------------------------------------------------------------------------------
def signal_func_1P(parameters_1P, m_star, times):
    """
    Calculate the signal function for a one-planet signal based on the parameters.

    Parameters:
    pars (tuple): A tuple containing the parameters needed for the signal calculation.
        - alpha0 (float): Right ascension of the source in degrees.
        - delta0 (float): Declination of the source in degrees.
        - mu_alpha (float): Proper motion in right ascension in milliarcseconds per year.
        - mu_delta (float): Proper motion in declination in milliarcseconds per year.
        - parallax (float): Parallax angle in milliarcseconds.
        - e (float): Eccentricity of the planet's orbit.
        - omega (float): Argument of periastron of the planet's orbit in radians.
        - Omega (float): Longitude of ascending node of the planet's orbit in radians.
        - cos_i (float): Cosine of the inclination angle of the planet's orbit.
        - log_m_planet (float): Log of the mass of the planet in log10 Jupiter masses.
        - log_P (float): Log of the orbital period of the planet in log10 years.
        - t_peri (float): Time of perigee passage of the planet in years.
        - m_star (float): Mass of the host star in solar masses.
    times (array-like): Array of time values.

    Returns:
    tuple: A tuple containing the signal in right ascension (signal_ra) and declination (signal_dec) directions.
    """
    
    
    # Unpack the parameters_1P array 
    alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, log_m_planet, log_P, t_peri = parameters_1P
    
    # Calculate the proper motion term 
    prop_ra, prop_dec = generate_pm_signal(mu_alpha, mu_delta, times)  # [uas]
    
    # Calculate the parallax term 
    parallax_ra, parallax_dec = generate_parallax_signal(alpha0, delta0, parallax, times)  # [uas]
    
    # Calculate the planetary term
    planetary_params = e, omega, Omega, cos_i, log_m_planet, log_P, t_peri
    planetary_ra, planetary_dec = generate_planet_signal(planetary_params, parallax, m_star, times)  # [uas]
    
    # Add all three components together 
    signal_ra  = prop_ra  + parallax_ra  + planetary_ra  # [uas]
    signal_dec = prop_dec + parallax_dec + planetary_dec # [uas]
    
    return signal_ra, signal_dec # [uas, uas]
# ---------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------
def find_signal_components(df, N_synthetic, N_model, print_params= True, print_alpha = True):  
    """
    Calculate various astrometric signal components and noise/error components.

    Args:
        df (DataFrame): The input data frame.
        N_synthetic (int): Number of time steps for synthetic data.
        N_model (int): Number of time steps for model data.
        print_params (bool, optional): Whether to print parameters. Defaults to True.
        print_alpha (bool, optional): Whether to print astrometric signature. Defaults to True.

    Returns:
        tuple: A tuple containing the following:
            - np_parameters (list): Parameters without a planet.
            - wp_parameters (list): Parameters with a planet.
            - m_star: Stellar mass.
            - synthetic_signal (list): Synthetic signal components and time.
            - model_signal (list): Model signal components and time.
            - error_components (list): Noise and error components.
            - alpha: Astrometric signature.
            - m_star: Stellar mass.
    """
    
    # Setting initial parameters:
    # ---------------------------------------------------------------------------
    n_object = 1 # Number of systems (only works for 1 right now) 
    
    # Make time arrays for synthetic and model data 
    # The length of the mission is hardcoded at 5 years 
    times_synthetic = np.linspace(0, 5, N_synthetic)  # Set the times array for the synthetic data using N_synthetic [years]
    times_model     = np.linspace(0, 5, N_model)      # Set the times array for the model data using N_model [years]
    # ---------------------------------------------------------------------------
    
    # Find planetary and Gaia parameters 
    # ---------------------------------------------------------------------------
    # Find the planetary parameters using the planetary_params function 
    # In planetary_params the period range is hardcoded to be (0.01, 10 years)
    # The units in order will be [unitless, rad, rad, unitless, log10(M_Jup), log10(years), years]
    e, omega, Omega, cos_i, log_m_planet, log_P, t_peri  = planetary_params(n_object) 
    
    
    # Find the Gaia parameters using the gaia_params function
    # The units in order will be [deg, deg, mas/year, mas.year, mas, M_sun]
    alpha0, delta0, mu_alpha, mu_delta, parallax, m_star, x = gaia_params(df, n_object)
    
    # Arrays holding parameter values 
    parameters_0P = [alpha0, delta0, mu_alpha, mu_delta, parallax]
    parameters_1P = [alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, log_m_planet, log_P, t_peri]
    # ---------------------------------------------------------------------------
    
    # Print out parameters
    # ---------------------------------------------------------------------------
    # based on the input of print_params the function will print out the parameters that planetary_params and gaia_params found
    
    # Make an array of the values for printing 
    planetary_values_for_printing = [e[0], omega[0], Omega[0], cos_i[0], log_m_planet[0], log_P[0], t_peri[0]]
    gaia_values_for_printing = [alpha0[0], delta0[0],mu_alpha[0], mu_delta[0], parallax[0], m_star[0]]
    
    # if print_params = True then print the parameters 
    if print_params:
        print_parameters(gaia_values_for_printing, planetary_values_for_printing)
        print("x              :", x[0])
    # ---------------------------------------------------------------------------  
        
        
    # Calculate the astrometric signature 
    # ---------------------------------------------------------------------------
    # Find distance d using the calculate_distance function 
    d_pc = calculate_distance(parallax) # [pc] 
    
    # Calculate the astrometric signature using the calculate_astrometric_signature function
    alpha_uas = calculate_astrometric_signature(log_m_planet, m_star, log_P, d_pc) # [uas]
    
    if print_alpha:
        print(" ")
        print('Astrometric signature:', alpha_uas[0], "[uas]")
    # ---------------------------------------------------------------------------
    
    # Find signal components in uas for synthetic and model 
    # ---------------------------------------------------------------------------
    # Proper motion signal 
    prop_ra_synthetic, prop_dec_synthetic = generate_pm_signal(mu_alpha, mu_delta, times_synthetic) # [uas]
    prop_ra_model, prop_dec_model         = generate_pm_signal(mu_alpha, mu_delta, times_model)     # [uas]
    
    # Parallax signal 
    parallax_ra_synthetic, parallax_dec_synthetic = generate_parallax_signal(alpha0, delta0, parallax, times_synthetic) # [uas]
    parallax_ra_model, parallax_dec_model         = generate_parallax_signal(alpha0, delta0, parallax, times_model)     # [uas]
    
    # Planet signal
    planetary_pars = e, omega, Omega, cos_i, log_m_planet, log_P, t_peri
    planetary_ra_synthetic, planetary_dec_synthetic = generate_planet_signal(planetary_pars, 
                                                                             parallax, 
                                                                             m_star, 
                                                                             times_synthetic) # [uas]
    
    planetary_ra_model, planetary_dec_model         = generate_planet_signal(planetary_pars, 
                                                                             parallax, 
                                                                             m_star, 
                                                                             times_model)     # [uas]
    # ---------------------------------------------------------------------------
    
    # Other stuff
    # ---------------------------------------------------------------------------
    # Find signal_fov 
    sigma_fov(df) # [uas]
    
    # Create noise in the RA and Dec directions 
    noise_ra  = np.random.normal(0, df.sigma_fov[x], N_synthetic)  # [uas]
    noise_dec = np.random.normal(0, df.sigma_fov[x], N_synthetic)  # [uas]
    
    # Create errors 
    errors = np.zeros(N_synthetic) + df.sigma_fov[x[0]] # [uas]
    # ---------------------------------------------------------------------------
    
    # Set up the return statement 
    # ---------------------------------------------------------------------------
    # Arrays holding synthetic signal components and time
    synthetic_signal = [prop_ra_synthetic, prop_dec_synthetic, parallax_ra_synthetic,
                         parallax_dec_synthetic, planetary_ra_synthetic, planetary_dec_synthetic, times_synthetic]
    
    # Arrays holding model signal components and time
    model_signal = [prop_ra_model, prop_dec_model, parallax_ra_model, parallax_dec_model, 
                     planetary_ra_model, planetary_dec_model, times_model]
    
    # Array holding noise and error components
    error_components = [noise_ra, noise_dec, errors]
    # ---------------------------------------------------------------------------
    
    return (parameters_0P, parameters_1P, synthetic_signal, model_signal, error_components, alpha_uas, m_star) 
# -----------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------
def HARDCODED_find_signal_components(df, N_synthetic, N_model, print_params= True, print_alpha = True):  
    """
    Calculate various astrometric signal components and noise/error components.

    Args:
        df (DataFrame): The input data frame.
        N_synthetic (int): Number of time steps for synthetic data.
        N_model (int): Number of time steps for model data.
        print_params (bool, optional): Whether to print parameters. Defaults to True.
        print_alpha (bool, optional): Whether to print astrometric signature. Defaults to True.

    Returns:
        tuple: A tuple containing the following:
            - np_parameters (list): Parameters without a planet.
            - wp_parameters (list): Parameters with a planet.
            - m_star: Stellar mass.
            - synthetic_signal (list): Synthetic signal components and time.
            - model_signal (list): Model signal components and time.
            - error_components (list): Noise and error components.
            - alpha: Astrometric signature.
            - m_star: Stellar mass.
    """
    
    # Setting initial parameters:
    # ---------------------------------------------------------------------------
    n_object = 1 # Number of systems (only works for 1 right now) 
    
    # Make time arrays for synthetic and model data 
    # The length of the mission is hardcoded at 5 years 
    times_synthetic = np.linspace(0, 5, N_synthetic)  # Set the times array for the synthetic data using N_synthetic [years]
    times_model     = np.linspace(0, 5, N_model)      # Set the times array for the model data using N_model [years]
    # ---------------------------------------------------------------------------
    
    # Find planetary and Gaia parameters 
    # ---------------------------------------------------------------------------
    # Find the planetary parameters using the planetary_params function 
    # In planetary_params the period range is hardcoded to be (0.01, 10 years)
    # The units in order will be [unitless, rad, rad, unitless, log10(M_Jup), log10(years), years]
    e, omega, Omega, cos_i, log_m_planet, log_P, t_peri  = HARDCODED_planetary_params(n_object) 
    
    
    # Find the Gaia parameters using the gaia_params function
    # The units in order will be [deg, deg, mas/year, mas.year, mas, M_sun]
    alpha0, delta0, mu_alpha, mu_delta, parallax, m_star, x = HARDCODED_gaia_params(df, n_object)
    
    # Arrays holding parameter values 
    parameters_0P = [alpha0, delta0, mu_alpha, mu_delta, parallax]
    parameters_1P = [alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, log_m_planet, log_P, t_peri]
    # ---------------------------------------------------------------------------
    
    # Print out parameters
    # ---------------------------------------------------------------------------
    # based on the input of print_params the function will print out the parameters that planetary_params and gaia_params found
    
    # Make an array of the values for printing 
    planetary_values_for_printing = [e[0], omega[0], Omega[0], cos_i[0], log_m_planet[0], log_P[0], t_peri[0]]
    gaia_values_for_printing = [alpha0[0], delta0[0],mu_alpha[0], mu_delta[0], parallax[0], m_star[0]]
    
    # if print_params = True then print the parameters 
    if print_params:
        print_parameters(gaia_values_for_printing, planetary_values_for_printing)
        print("x              :", x[0])
    # ---------------------------------------------------------------------------  
        
        
    # Calculate the astrometric signature 
    # ---------------------------------------------------------------------------
    # Find distance d using the calculate_distance function 
    d_pc = calculate_distance(parallax) # [pc] 
    
    # Calculate the astrometric signature using the calculate_astrometric_signature function
    alpha_uas = calculate_astrometric_signature(log_m_planet, m_star, log_P, d_pc) # [uas]
    
    if print_alpha:
        print(" ")
        print('Astrometric signature:', alpha_uas[0], "[uas]")
    # ---------------------------------------------------------------------------
    
    # Find signal components in uas for synthetic and model 
    # ---------------------------------------------------------------------------
    # Proper motion signal 
    prop_ra_synthetic, prop_dec_synthetic = generate_pm_signal(mu_alpha, mu_delta, times_synthetic) # [uas]
    prop_ra_model, prop_dec_model         = generate_pm_signal(mu_alpha, mu_delta, times_model)     # [uas]
    
    # Parallax signal 
    parallax_ra_synthetic, parallax_dec_synthetic = generate_parallax_signal(alpha0, delta0, parallax, times_synthetic) # [uas]
    parallax_ra_model, parallax_dec_model         = generate_parallax_signal(alpha0, delta0, parallax, times_model)     # [uas]
    
    # Planet signal
    planetary_pars = e, omega, Omega, cos_i, log_m_planet, log_P, t_peri
    planetary_ra_synthetic, planetary_dec_synthetic = generate_planet_signal(planetary_pars, 
                                                                             parallax, 
                                                                             m_star, 
                                                                             times_synthetic) # [uas]
    
    planetary_ra_model, planetary_dec_model         = generate_planet_signal(planetary_pars, 
                                                                             parallax, 
                                                                             m_star, 
                                                                             times_model)     # [uas]
    # ---------------------------------------------------------------------------
    
    # Other stuff
    # ---------------------------------------------------------------------------
    # Find signal_fov 
    sigma_fov(df) # [uas]
    
    # Create noise in the RA and Dec directions 
    noise_ra  = np.random.normal(0, df.sigma_fov[x], N_synthetic)  # [uas]
    noise_dec = np.random.normal(0, df.sigma_fov[x], N_synthetic)  # [uas]
    
    # Create errors 
    errors = np.zeros(N_synthetic) + df.sigma_fov[x[0]] # [uas]
    # ---------------------------------------------------------------------------
    
    # Set up the return statement 
    # ---------------------------------------------------------------------------
    # Arrays holding synthetic signal components and time
    synthetic_signal = [prop_ra_synthetic, prop_dec_synthetic, parallax_ra_synthetic,
                         parallax_dec_synthetic, planetary_ra_synthetic, planetary_dec_synthetic, times_synthetic]
    
    # Arrays holding model signal components and time
    model_signal = [prop_ra_model, prop_dec_model, parallax_ra_model, parallax_dec_model, 
                     planetary_ra_model, planetary_dec_model, times_model]
    
    # Array holding noise and error components
    error_components = [noise_ra, noise_dec, errors]
    # ---------------------------------------------------------------------------
    
    return (parameters_0P, parameters_1P, synthetic_signal, model_signal, error_components, alpha_uas, m_star) 
# -----------------------------------------------------------------------------------------------------------------------------
