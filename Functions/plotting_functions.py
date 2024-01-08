# This python file contains all unit conversion functions

import numpy as np
from numpy import cos, sin
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