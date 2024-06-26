# This python file contains contants 
import numpy as np

def print_parameters(gaia_values , planetary_values):
    planetary_names = ['e', 'omega', 'Omega', 'cos_i', 'log10(m_planet)', 'log10(P_orb)', 't_peri']
    planetary_units = ['unitless', 'radians', 'radians', 'unitless', 'log10(Jupiter masses)', 'log10(years)', 'years']
    
    gaia_names = ['alpha0', 'delta0', 'mu_alpha', 'mu_delta', 'parallax', 'm_star']
    gaia_units = ['degrees', 'degrees', 'mas/year', 'mas/year', 'mas', 'M_sun', '(array)']
    
    max_name_length = max(max(map(len, planetary_names)), max(map(len, gaia_names)))

    print("Planetary Parameters:")
    for name, value, unit in zip(planetary_names, planetary_values, planetary_units):
        print(f"{name.ljust(max_name_length)}: {value}  [{unit}]")
    
    print(" ")
    print("Gaia Parameters:")
    for name, value, unit in zip(gaia_names, gaia_values, gaia_units):
        print(f"{name.ljust(max_name_length)}: {value}  [{unit}]")  
        

        
def print_parameter_differences(true_parameters, fitted_parameters, return_type):
    # Define np_parameter_names and parameter_names within the function
    np_parameter_names = ['alpha0', 'delta0', 'mu_alpha', 'mu_delta', 'parallax']
    wp_parameter_names = ['alpha0', 'delta0', 'mu_alpha', 'mu_delta', 'parallax','e', 'omega', 'Omega', 'cos_i', 'log10(m_planet)', 'log10(P_orb)', 't_peri']

    # Use the defined names within the function
    if return_type == 'np':
        max_width = max(len(name) for name in np_parameter_names)
        for name, real_value, fitted_value in zip(np_parameter_names, true_parameters, fitted_parameters):
            difference = real_value - fitted_value
            print(f'{name: <{max_width}}: real={real_value.item():.5f}, fitted={fitted_value.item():.5f}, difference={np.abs(difference.item()):.5f}')
    elif return_type == 'wp':
        max_width = max(len(name) for name in wp_parameter_names)
        for name, real_value, fitted_value in zip(wp_parameter_names, true_parameters, fitted_parameters):
            difference = real_value - fitted_value
            print(f'{name: <{max_width}}: real={real_value.item():.5f}, fitted={fitted_value.item():.5f}, difference={np.abs(difference.item()):.5f}')
    else:
        raise ValueError("Invalid return_type. Use 'np' or 'wp'.")






