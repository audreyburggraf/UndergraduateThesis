# This python file contains all unit conversion functions

import numpy as np

# mass
# -----------------------------------------------------------------------------------------------------------------------------
def solar_to_jupiter_mass(mass_in_solar):
    """
    Convert mass from solar masses to Jupiter masses.

    Parameters:
    - mass_in_solar (float): Mass in solar masses.

    Returns:
    - float: Mass in Jupiter masses.
    """
    # Conversion factor: 1 solar mass = approximately 1047.57 Jupiter masses
    jupiter_mass = mass_in_solar * 1047.57
    return jupiter_mass
# -----------------------------------------------------------------------------------------------------------------------------
def jupiter_to_solar_mass(mass_in_jupiter):
    """
    Convert mass from Jupiter masses to solar masses.

    Parameters:
    - mass_in_jupiter (float): Mass in Jupiter masses.

    Returns:
    - float: Mass in solar masses.
    """
    # Conversion factor: 1 solar mass = approximately 1047.57 Jupiter masses
    solar_mass = mass_in_jupiter / 1047.57
    return solar_mass
# -----------------------------------------------------------------------------------------------------------------------------
def earth_to_jupiter_mass(mass_in_earth):
    """
    Convert mass from earth masses to Jupiter masses.

    Parameters:
    - mass_in_earth (float): Mass in earth masses.

    Returns:
    - float: Mass in Jupiter masses.
    """
    # Conversion factor: 1 earth mass = approximately 0.00314558 Jupiter masses
    jupiter_mass = mass_in_earth * 0.00314558
    return jupiter_mass
# -----------------------------------------------------------------------------------------------------------------------------

# angular 
# -----------------------------------------------------------------------------------------------------------------------------
def radians_to_microarcseconds(radians):
    """
    Convert radians to microarcseconds.

    Parameters:
    - radians (float): Angle in radians.

    Returns:
    float: Angle converted to microarcseconds.
    """
    microarcseconds_per_radian = 1e6 * 180 * 3600 / np.pi
    microarcseconds = radians * microarcseconds_per_radian
    return microarcseconds
# -----------------------------------------------------------------------------------------------------------------------------
def radians_to_milliarcseconds(radians):
    """
    Convert radians to milliarcseconds.

    Parameters:
    - radians (float): Angle in radians.

    Returns:
    float: Angle converted to milliarcseconds.
    """
    milliarcseconds_per_radian = 180 * 3600 * 1e3 / np.pi
    milliarcseconds = radians * milliarcseconds_per_radian
    return milliarcseconds
# -----------------------------------------------------------------------------------------------------------------------------
def arcseconds_to_milliarcseconds(arcseconds):
    """
    Convert arcseconds to milliarcseconds.

    Parameters:
    - arcseconds (float): Angle in arcseconds.

    Returns:
    float: Angle converted to milliarcseconds.
    """
    milliarcseconds_per_arcsecond = 1e3
    milliarcseconds = arcseconds * milliarcseconds_per_arcsecond
    return milliarcseconds
# -----------------------------------------------------------------------------------------------------------------------------
def arcseconds_to_microarcseconds(arcseconds):
    """
    Convert arcseconds to microarcseconds.

    Parameters:
    - arcseconds (float): Angle in arcseconds.

    Returns:
    float: Angle converted to microarcseconds.
    """
    microarcseconds_per_arcsecond = 1e6
    microarcseconds = arcseconds * microarcseconds_per_arcsecond
    return microarcseconds
# -----------------------------------------------------------------------------------------------------------------------------
def microarcseconds_to_radians(microarcseconds):
    """
    Convert microarcseconds to radians.

    Parameters:
    - microarcseconds (float): Angle in microarcseconds.

    Returns:
    float: Angle converted to radians.
    """
    radians_per_microarcsecond = np.pi / (180 * 3600 * 1e6)
    radians = microarcseconds * radians_per_microarcsecond
    return radians
# -----------------------------------------------------------------------------------------------------------------------------
def milliarcseconds_to_radians(milliarcseconds):
    """
    Convert milliarcseconds to radians.

    Parameters:
    - milliarcseconds (float): Angle in milliarcseconds.

    Returns:
    float: Angle converted to radians.
    """
    radians_per_milliarcsecond = np.pi / (180 * 3600 * 1e3)
    radians = milliarcseconds * radians_per_milliarcsecond
    return radians
# -----------------------------------------------------------------------------------------------------------------------------
def milliarcseconds_to_arcseconds(milliarcseconds):
    """
    Convert milliarcseconds to arcseconds.

    Parameters:
    - milliarcseconds (float): Angle in milliarcseconds.

    Returns:
    float: Angle converted to arcseconds.
    """
    arcseconds_per_milliarcsecond = 1 / 1000.0
    arcseconds = milliarcseconds * arcseconds_per_milliarcsecond
    return arcseconds
# -----------------------------------------------------------------------------------------------------------------------------
def milliarcseconds_to_microarcseconds(distance_milliarcsec):
    """
    Convert distance from milliarcseconds to microarcseconds.

    Parameters:
    distance_milliarcsec (float): Distance in milliarcseconds.

    Returns:
    float: Distance in microarcseconds.
    """
    # Convert milliarcseconds to microarcseconds
    distance_microarcsec = distance_milliarcsec * 1e3

    return distance_microarcsec
# -----------------------------------------------------------------------------------------------------------------------------
def degrees_to_radians(degrees):
    """
    Convert degrees to radians.

    Parameters:
    - degrees (float): Angle in degrees.

    Returns:
    float: Angle converted to radians.
    """
    radians_per_degree = np.pi / 180
    radians = degrees * radians_per_degree
    return radians
# -----------------------------------------------------------------------------------------------------------------------------
def degrees_to_milliarcseconds(degrees):
    """
    Convert degrees to milliarcseconds.

    Parameters:
    - degrees (float): Angle in degrees.

    Returns:
    float: Angle converted to milliarcseconds.
    """
    milliarcseconds_per_degree = 3600 * 1e3
    milliarcseconds = degrees * milliarcseconds_per_degree
    return milliarcseconds
# -----------------------------------------------------------------------------------------------------------------------------
def au_to_microarcseconds(distance_au, distance_pc):
    """
    Convert distance from AU to microarcseconds.

    Parameters:
    distance_au (float): Distance in Astronomical Units.
    distance_pc (float): Distance to the observer in parsecs.

    Returns:
    float: Distance in microarcseconds.
    """
    # Convert AU to arcseconds
    angular_distance_arcsec = distance_au / distance_pc

    # Convert arcseconds to microarcseconds
    angular_distance_microarcsec = angular_distance_arcsec * 1e6

    return angular_distance_microarcsec
# -----------------------------------------------------------------------------------------------------------------------------


# time 
# -----------------------------------------------------------------------------------------------------------------------------
def days_to_years(time_in_days):
    """
    Convert time in days to years.

    Parameters:
    - time_in_days (float): Time in days.

    Returns:
    float: Time converted to years.
    """
    days_per_year = 365.25  # Taking into account leap years
    years = time_in_days / days_per_year
    return years
# -----------------------------------------------------------------------------------------------------------------------------