# This python file contains all the functions needed for the signal files to work 

import numpy as np
from numpy import cos, sin

def calculate_thiele_innes(omega, Omega, cos_i, parallax, m_planet, P_orb, m_star=1, G_const=4*np.pi**2):
    n = 2*np.pi/P_orb

    a_AU = (G_const*(m_star+m_planet)/n**2)**(1/3)

    a_hat = (m_planet/(m_star+m_planet))*a_AU*parallax

    sin_i = np.sqrt(1-cos_i**2)

    A = a_hat * (  cos(omega)  * cos(Omega) - sin(omega) * sin(Omega) * cos_i)  
    F = a_hat * ( - sin(omega) * cos(Omega) - cos(omega) * sin(Omega) * cos_i)

    B = a_hat * (  cos(omega)  * sin(Omega) + sin(omega) * cos(Omega) * cos_i)  
    G = a_hat * ( - sin(omega) * sin(Omega) + cos(omega) * cos(Omega) * cos_i)

    C = a_hat * sin(omega)*sin_i
    H = a_hat * sin_i*cos(omega)

    return(B, A, F, G, H, C, a_hat)