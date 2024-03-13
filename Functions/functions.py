# This python file contains all the functions needed for the signal files to work 

import numpy as np
from numpy import cos, sin
import rebound
from scipy.interpolate import interp1d

# import unit conversions 
from unit_conversion_functions import *
from printing_functions import *


# hardcoded planetary functions 
# -----------------------------------------------------------------------------------------------------------------------------
def HARDCODED_planetary_params(N):
    """
    Hardcode in planetary parameters to check the fitting functions 
    We want a planet that will for sure be detected 
    
    Generate random orbital parameters and characteristics for a specified number of planets.

    Parameters:
    N (int): Number of planets.

    Returns:
    tuple: A tuple containing the following arrays for N planets:
        - e (numpy.ndarray): Eccentricity of the orbits (unitless).
        - omega (numpy.ndarray): Argument of periastron in radians.
        - Omega (numpy.ndarray): Longitude of ascending node in radians.
        - cos_i (numpy.ndarray): Cosine of the orbital inclination angle (unitless).
        - log_m_planet (numpy.ndarray): Log of mass of the planets in log10 Jupiter masses log10(M_Jupiter).
        - log_P (numpy.ndarray): Log of orbital period of the planets in log10(years).
        - t_peri (numpy.ndarray): Time of perigee passage in years.
    """
    # Setting seed to keep parameters constant that don't have a set value 
    np.random.seed(5) 
    
    # Orbital parameters
    e = np.random.uniform(0, 0.5, N)           # [unitless]
    omega = np.random.uniform(0, 2*np.pi, N)   # [radians]
    Omega = np.random.uniform(0, 2*np.pi, N)   # [radians]
    cos_i = np.random.uniform(0, 1, N)         # [unitless]

    # Planet mass
    log_m_planet = np.array([np.log10(10)])         # [log10(M_Jupiter)]

    # Period and time of perigee passage
    P = np.array([2])                               # period is 2 years [years]
    log_P = np.log10(P)                 # log period is log10(2 years) [log10(years)]
    t_peri = np.random.uniform(0, P, N) # time of perigee passage is between 0 and 2 years [years]

    return e, omega, Omega, cos_i, log_m_planet, log_P, t_peri
# -----------------------------------------------------------------------------------------------------------------------------
# UPDATE DOCTRING
def HARDCODED_gaia_params(df, N):
    """
    Hardcode the gaia parameters to check the fitting functions 
    We want a bright star
    
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
    # Setting seed to keep parameters constant that don't have a set value
    np.random.seed(5) 
    
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






# parameter functions 
# -----------------------------------------------------------------------------------------------------------------------------
def planetary_params(N):
    """
    Generate random orbital parameters and characteristics for a specified number of planets.

    Parameters:
    N (int): Number of planets.

    Returns:
    tuple: A tuple containing the following arrays for N planets:
        - e (numpy.ndarray): Eccentricity of the orbits (unitless).
        - omega (numpy.ndarray): Argument of periastron in radians.
        - Omega (numpy.ndarray): Longitude of ascending node in radians.
        - cos_i (numpy.ndarray): Cosine of the orbital inclination angle (unitless).
        - log_m_planet (numpy.ndarray): Log of mass of the planets in log10 Jupiter masses log10(M_Jupiter).
        - log_P (numpy.ndarray): Log of orbital period of the planets in log10(years).
        - t_peri (numpy.ndarray): Time of perigee passage in years.
    """

    # np.random.seed(5) # debugging
    
    # Orbital parameters
    e = np.random.uniform(0, 0.5, N)           # [unitless]
    omega = np.random.uniform(0, 2*np.pi, N)   # [radians]
    Omega = np.random.uniform(0, 2*np.pi, N)   # [radians]
    cos_i = np.random.uniform(0, 1, N)         # [unitless]

    # Planet mass
    log_m_planet = np.random.uniform(np.log10(0.3), np.log10(13), N)   # [log10(M_Jupiter)]

    # Period and time of perigee passage
    # Period range is hardcoded as 0.01 to 10 years 
    log_P = np.random.uniform(np.log10(0.01), np.log10(10), N)       # [log10(years)]
    
    P = 10**log_P                              # [years]
    t_peri = np.random.uniform(0, P, N)        # [years]

    return e, omega, Omega, cos_i, log_m_planet, log_P, t_peri
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
    # np.random.seed(5) # debugging
    
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
def calculate_astrometric_signature(log_m_planet, m_star, log_P, d):
    """
    Calculate the astrometric signature (angular displacement) of a planet orbiting a star.

    This function calculates the angular displacement caused by a planet's motion around a star, 
    observable from Earth, known as the astrometric signature. The astrometric signature is measured
    in microarcseconds (uas) and represents the apparent change in position of the star due to the 
    gravitational influence of the orbiting planet.

    Parameters:
        log_m_planet (float): Log10 of the mass of the planet in log10(Jupiter masses)
        m_star (float): Mass of the star in solar masses.
        log_P (float): Log10 of the orbital period of the planet in years.
        d (float): Distance between the star and observer in parsecs.

    Returns:
        float: The astrometric signature (angular displacement) of the planet in microarcseconds (uas).
    """
    
    # Convert log planet mass from log Jupiter masses to solar masses
    m_planet_solar = jupiter_to_solar_mass(10**log_m_planet) # [M_sun]
    
    # Calculate the semi-major axis of the planet's orbit in astronomical units (AU)
    a = calculate_semi_major_axis(log_P, m_star)  # [AU]

    # Calculate the astrometric signature (alpha) in arcseconds
    alpha_arcsec = (m_planet_solar / m_star) * a * (d**(-1)) # [arcseconds]
    
    # Convert the astrometric signature from arcseconds to microarcseconds
    alpha_uas = arcseconds_to_microarcseconds(alpha_arcsec) # [uas]

    return alpha_uas

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

# import warnings, pdb
# warnings.filterwarnings('error')

def calculate_semi_major_axis(log_P, m_star):
    """
    Calculate the semi-major axis using the provided formula.

    Parameters:
    - log_P (float): Orbital period of the planet in log10(years).
    - m_star (float): Mass of the star in solar masses.

    Returns:
    - tuple: Semi-major axis calculated based on the input parameters, in astronomical units (AU),
             along with a string indicating any issues with the inputs.
    """
    

    # Change log10(P) to P [log10(years) to years]
    P_orb = 10**log_P

    a = (P_orb**2 * m_star)**(1/3)  # [AU] 
    
    return a

    
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
    return distance_pc # [pc]
# -----------------------------------------------------------------------------------------------------------------------------

