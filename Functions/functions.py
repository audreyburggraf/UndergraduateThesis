# This python file contains all the functions needed for the signal files to work 

import numpy as np
from numpy import cos, sin
import rebound
from scipy.interpolate import interp1d

# import unit conversions 
from unit_conversion_functions import *
from printing_functions import *


# units
# -----------------------------------------------------------------------------------------------------------------------------
# GAIA PARAMETERS:
# alpha0        = [degrees]
# delta0        = [degrees]
# mu_alpha      = [mas/year]
# mu_delta      = [mas/year]
# parallax      = [mas]
# m_star        = [Solar masses]

# PLANETARY PARAMETERS:
# e        = [unitless]
# omega    = [radians] 
# Omega    = [radians] 
# cos_i    = [unitless]
# m_planet = [Jupiter masses] 
# log_P    = [log10(years)]
# t_peri   = [years]


# t = [years]
# alpha (astrometric signature) = [uas]
# signals = [uas] (proper motion, parallax, planetary)
# -----------------------------------------------------------------------------------------------------------------------------


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
def calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, m_star, log_P):
    """
    Calculate the Thiele-Innes constants for a binary star system with an orbiting planet.

    This function takes orbital parameters and masses of the star and planet as input and
    computes the Thiele-Innes constants (A, B, F, G, H, C, and semi-major axis) for the system.

    Parameters:
    omega (float): Argument of periastron in radians.
    Omega (float): Longitude of ascending node in radians.
    cos_i (float): Cosine of the orbital inclination angle (unitless).
    parallax (float): Parallax of the system in milliarcseconds (mas).
    m_planet (float): Mass of the planet in Jupiter masses (Mjupiter).
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
    alpha = astrometric_signature(m_planet, m_star, log_P, d) # [mas]
    
    # Find sin(inclination)
    sin_i = np.sqrt(1 - cos_i**2)  # [unitless]
    
    A = alpha * (cos(omega) * cos(Omega) - sin(omega) * sin(Omega) * cos_i)   # [uas]
    F = alpha * (-sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i)  # [uas]

    B = alpha * (cos(omega) * sin(Omega) + sin(omega) * cos(Omega) * cos_i)   # [uas]
    G = alpha * (-sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i)  # [uas]

    C = alpha * sin(omega) * sin_i  # [uas]
    H = alpha * cos(omega) * sin_i  # [uas]
    
    
    return B, A, F, G, H, C
# -----------------------------------------------------------------------------------------------------------------------------
# SWITCH TO LOG G

def generate_planet_signal(parallax, e, omega, Omega, cos_i, m_planet, m_star, log_P, t_peri, times):
    """
    Generates the astrometric signal of a planet orbiting a star.

    Parameters:
    - parallax_mas (float): Parallax in milliarcseconds (mas).
    - e (float): Orbital eccentricity (unitless).
    - omega (float): Argument of periastron in radians.
    - Omega (float): Longitude of the ascending node in radians.
    - cos_i (float): Cosine of the inclination angle (unitless).
    - LOG m_planet (float): Mass of the planet in Jupiter masses (Mjupiter).
    - m_star (float): Mass of the star in Solar masses (Msun).
    - log_P (float): Orbital period of the planet in log10(years).
    - t_peri (float): Time of periastron passage in years.
    - times (array-like): Array of time values at which to compute the signal in years.

    Returns:
    - plnt_term_ra (array): Astrometric signal in right ascension (microarcseconds).
    - plnt_term_dec (array): Astrometric signal in declination (microarcseconds).
    """
    
    # Thiele-Innes constants 
    B, A, F, G, _, _ = calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, m_star, log_P) # [uas]
    
    # find P_orb in years from log10(P) in log10(years)
    P_orb = 10**log_P                           # [years]
        
    # Anomalies?
    M = (2*np.pi)*(times - t_peri)/P_orb   # [radians]
    E = np.vectorize(rebound.M_to_E)(e,M)  # [radians]

    X = (cos(E)-e)               # [unitless]
    Y = np.sqrt((1-e**2))*sin(E) # [unitless]
     
    plnt_term_ra  = B*X + G*Y  # [mas] * [unitless] = [uas]
    plnt_term_dec = A*X + F*Y  # [mas] * [unitless] = [uas]

    return plnt_term_ra, plnt_term_dec
# ----------------------------------------------------------------------------------------------------------------------------

# parameter functions 
# -----------------------------------------------------------------------------------------------------------------------------
def planetary_params(N, min_P, max_P):
    """
    Generate random orbital parameters and characteristics for a specified number of planets.

    Parameters:
    N (int): Number of planets.
    min_P (float): Minimum orbital period in years.
    max_P (float): Maximum orbital period in years.

    Returns:
    tuple: A tuple containing the following arrays for N planets:
        - e (numpy.ndarray): Eccentricity of the orbits (unitless).
        - omega (numpy.ndarray): Argument of periastron in radians.
        - Omega (numpy.ndarray): Longitude of ascending node in radians.
        - cos_i (numpy.ndarray): Cosine of the orbital inclination angle (unitless).
        - m_planet (numpy.ndarray): Mass of the planets in Jupiter masses (M_Jupiter).
        - log_P (numpy.ndarray): Orbital period of the planets in log10(years).
        - t_peri (numpy.ndarray): Time of perigee passage in years.
    """
    # Setting random seed
    # np.random.seed(0)

    # Orbital parameters
    e = np.random.uniform(0, 0.5, N)           # [unitless]
    omega = np.random.uniform(0, 2*np.pi, N)   # [radians]
    Omega = np.random.uniform(0, 2*np.pi, N)   # [radians]
    cos_i = np.random.uniform(0, 1, N)         # [unitless]

    # Planet mass
    m_planet = 10**np.random.uniform(np.log10(0.3), np.log10(13), N)   # [M_Jupiter]

    # Period and time of perigee passage
    p1 = np.log10(min_P)
    p2 = np.log10(max_P)
    log_P = np.random.uniform(p1, p2, N)       # [log10(years)]
    
    P = 10**log_P                              # [years]
    t_peri = np.random.uniform(0, P, N)        # [years]

    return e, omega, Omega, cos_i, m_planet, log_P, t_peri
# -----------------------------------------------------------------------------------------------------------------------------
def gaia_params(df, N):
    """
    Randomly select N stars from Gaia data and extract relevant parameters.

    Parameters:
    df (pandas.DataFrame): Gaia data containing astrometric and stellar information.
    N (int): Number of stars to randomly select.

    Returns:
    tuple: A tuple containing arrays with parameters for N stars:
        - alpha0 (numpy.ndarray): Right ascension in degrees (deg).
        - delta0 (numpy.ndarray): Declination in degrees (deg).
        - mu_alpha (numpy.ndarray): Proper motion in right ascension in milliarcseconds per year (mas/year).
        - mu_delta (numpy.ndarray): Proper motion in declination in milliarcseconds per year (mas/year).
        - parallax (numpy.ndarray): Parallax of the stars in milliarcseconds (mas).
        - m_star (numpy.ndarray): Stellar mass of the stars in solar masses (M_sun).
        - x (numpy.ndarray): Indices of the selected stars in the original DataFrame.
    """

    # Randomly choose N stars
    x = np.random.randint(0, len(df), N)
    
    # Extracting astrometric and stellar parameters
    alpha0 = df.ra[x]                  # [degrees]
    delta0 = df.dec[x]                 # [degrees]
    
    mu_alpha = df.pmra[x]              # [mas]                        
    mu_delta = df.pmdec[x]             # [mas]                  
    parallax = df.parallax[x]          # [mas]
    m_star = df.stellar_mass[x]        # [M_sun]
    
    #return alpha0, delta0, Delta_alpha_0, Delta_delta_0, mu_alpha, mu_delta, parallax, m_star, x 
    return alpha0.values, delta0.values, mu_alpha.values, mu_delta.values, parallax.values, m_star.values, x

# -----------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------------
# THIS WILL ONLY WORK FOR N_OBJECT = 1 FOR NOW
# ADD UNITS 

def find_signal_components(df, N_synthetic, N_model, print_params='neither', print_other=False):  
    # Setting initial parameters:
    # ---------------------------------------------------------------------------
    n_object = 1                  # number of objects 
    
    if print_other:
        print("N for synthetic data: ", N_synthetic) # print out N_synthetic, the number of timesteps for the synthetic data(small)
        print("N for model data    : ", N_model)     # print out N_model, the number of timesteps for the synthetic data (larger)
        print(" ")
    
    # set time array for the synthetic and model data 
    times_synthetic = np.linspace(0, 5, N_synthetic)  # Set the times array for the synthetic data using N [years]
    times_model     = np.linspace(0, 5, N_model)      # Set the times array for the model data using N [years]
    # ---------------------------------------------------------------------------
    
    # Find planetary and Gaia parameters 
    # ---------------------------------------------------------------------------
    # planetary parameters [unitless, rad, rad, unitless, M_Jup, log10(years), years]
    e, omega, Omega, cos_i, m_planet, log_P, t_peri  = planetary_params(n_object, 0.1, 10) # 0.1 and 10 are in years
    
    # Gaia parameters [deg, deg, mas/year, mas.year, mas, M_sun]
    alpha0, delta0, mu_alpha, mu_delta, parallax, m_star, x = gaia_params(df, n_object)
    
    # RESETTING PARAMETERS TO COMPARE
    alpha0, delta0, parallax = np.array([1]), np.array([2]), np.array([50]) # degrees, degrees, mas
    
    # Print out parameters
    planetary_values = [e[0], omega[0], Omega[0], cos_i[0], m_planet[0], log_P[0], t_peri[0]]
    
    # Define name and units for each Gaia parameter
    gaia_values = [alpha0[0], delta0[0],mu_alpha[0], mu_delta[0], parallax[0], m_star[0]]
    
    # Print out value and its unit
    if print_params in ['both', 'planet']:
        print_parameters(planetary_values, 'planetary')

    if print_params in ['both', 'gaia']:
        print(" ")
        print_parameters(gaia_values, 'gaia')
        print("x = ", x[0])
        
        
        
    # ---------------------------------------------------------------------------
    
    # Calculate the astrometric signature to check
    # ---------------------------------------------------------------------------
    d = calculate_distance(parallax)
    alpha = astrometric_signature(m_planet, m_star, log_P, d)
    if print_other:
        print(" ")
        print("astrometric_signature(", m_planet, m_star, log_P, d,") = ", alpha, "[uas]")
    # ---------------------------------------------------------------------------
    
    # Find signal components
    # ---------------------------------------------------------------------------
    # Proper motion signal 
    prop_ra_synthetic, prop_dec_synthetic = generate_pm_signal(mu_alpha, mu_delta, times_synthetic) # [uas]
    prop_ra_model, prop_dec_model = generate_pm_signal(mu_alpha, mu_delta, times_model) # [uas]
    
    # Parallax signal 
    parallax_ra_synthetic, parallax_dec_synthetic = generate_parallax_signal(alpha0, delta0, parallax, times_synthetic)# [uas]
    parallax_ra_model, parallax_dec_model = generate_parallax_signal(alpha0, delta0, parallax, times_model)# [uas]
    
    # Planet signal
    planetary_pars = parallax, e, omega, Omega, cos_i, np.log10(m_planet), m_star, log_P, t_peri
    planetary_ra_synthetic, planetary_dec_synthetic = generate_planet_signal(*planetary_pars, times_synthetic) # [uas]
    planetary_ra_model, planetary_dec_model = generate_planet_signal(*planetary_pars, times_model) # [uas]
    # ---------------------------------------------------------------------------
    
    # Step 5: find signal_fov 
    sigma_fov(df) # [uas]
    
    # Step 6: Create noise 
    noise_ra = np.random.normal(0, df.sigma_fov[x], N_synthetic)  # [uas]
    noise_dec = np.random.normal(0, df.sigma_fov[x], N_synthetic) # [uas]
    
    # Step 7: Create errors 

    errors = np.zeros(N_synthetic) + df.sigma_fov[x[0]] # [uas]
    
    # Break down return statement 
    # ---------------------------------------------------------------------------
    # setting the no planet and 1 planet parameter arrays in their proper units 
    np_parameters = [alpha0, delta0, mu_alpha, mu_delta, parallax]
    parameters = [alpha0, delta0, mu_alpha, mu_delta, parallax, e, omega, Omega, cos_i, m_planet, log_P, t_peri, m_star]
    
    # parameter_result includes the no planet and 1 planet parameters in their proper units 
    parameter_result = [np_parameters, parameters]
    
    # model_results includes the signal components and times for the synthetic in [uas] and [years]
    synthetic_result = [prop_ra_synthetic, prop_dec_synthetic, parallax_ra_synthetic, parallax_dec_synthetic, planetary_ra_synthetic, planetary_dec_synthetic, times_synthetic]
    
    # model_results includes the signal components and times for the model in [uas] and [years]
    model_result = [prop_ra_model, prop_dec_model, parallax_ra_model, parallax_dec_model, planetary_ra_model, planetary_dec_model, times_model]
    
    # error_result includes noise in ra and dec direction plus error (same for ra and dec) in [uas]
    error_result = [noise_ra, noise_dec, errors]
    # ---------------------------------------------------------------------------
    
    # return statement 
    return (parameter_result, synthetic_result, model_result, error_result) 
# -----------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------------
def astrometric_signature(m_planet, m_star, log_P, d):
    """
    Calculate the astrometric signature of a planet orbiting a star.

    This function calculates the astrometric signature (angular displacement) of a planet
    orbiting a star based on the given parameters.

    Parameters:
    m_planet (float): Mass of the planet in jupiter masses.
    m_star (float): Mass of the star in solar masses.
    log_P (float): Orbital period of the planet in log10(years).
    d (float): Distance between the star and observer in parsecs.

    Returns:
    float: The astrometric signature (angular displacement) of the planet in microarcseconds
    """
    
    # change planet mass to solar masses 
    m_planet = jupiter_to_solar_mass(m_planet) # [M_sun] from [Jupiter masses]
    
    # Calculate the semi-major axis of the planet's orbit in AU
    a = calculate_semi_major_axis(log_P, m_star)  # [AU]


    # Calculate the astrometric signature (alpha) in arcseconds
    alpha = (m_planet / m_star) * a * (d**(-1)) # [arcseconds]
    
    # convert it to microarcseconds
    alpha = arcseconds_to_microarcseconds(alpha) # [uas] from [arcseconds]

    return alpha
# -----------------------------------------------------------------------------------------------------------------------------
def find_star_mass(M_ks):
    """
    Calculate mass based on the Ks-band absolute magnitude (M_ks) using a polynomial fit.

    Parameters:
    M_ks (float): The Ks-band absolute magnitude of the object.

    Returns:
    float: The estimated mass of the object based on the provided Ks-band absolute magnitude.
    """
    f = np.poly1d([-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4][::-1])
    
    return 10**f(M_ks-7.5)
# -----------------------------------------------------------------------------------------------------------------------------
# This function uses data from (paper) and interp1d to add a sigma_fov column to the df in uas 
def sigma_fov(df):
    G = range(6,21)
    sigma_fov_interp = [34.2, 34.2, 34.2, 34.2, 34.2, 34.2, 34.2, 41.6, 56.1, 82.0, 125.7, 198.3, 320.6, 538.4, 996.5]

    interp_sigma = interp1d(G, sigma_fov_interp, bounds_error=False, fill_value=(34.2, np.nan))

    df["sigma_fov"] = interp_sigma(df.phot_g_mean_mag) # uas
# -----------------------------------------------------------------------------------------------------------------------------

def calculate_semi_major_axis(log_P, m_star):
    """
    Calculate the semi-major axis using the provided formula.

    Parameters:
    - log_P (float): Orbital period of the planet in log10(years).
    - m_star (float): Mass of the star in solar masses.

    Returns:
    - float: Semi-major axis calculated based on the input parameters, in astronomical units (AU).
    """
    
    # Change log10(P) to P [log10(years) to years]
    P_orb = 10**log_P
    
    try:
        a = (P_orb**2 * m_star)**(1/3)  # [AU]
    except Exception as e:
        print("Error:", e)
        print("P_orb:", P_orb)
        print("m_star:", m_star)
        raise  # re-raise the exception to see the traceback
    
    return a
    
    
#     a = (P_orb**2 * m_star)**(1/3) # [AU]
    
    
#     return a
# -----------------------------------------------------------------------------------------------------------------------------
def calculate_distance(parallax_mas):
    """
    Find distance from parallax. 

    Parameters:
    - parallax_mas (float): Parallax in milliarcseconds.

    Returns:
    float: Distance in parsecs.
    """
    parallax_arcseconds = milliarcseconds_to_arcseconds(parallax_mas) # [arcsconds] from [mas]
    
    distance_pc = 1 / parallax_arcseconds # [pc]
    return distance_pc
# -----------------------------------------------------------------------------------------------------------------------------

