import numpy as np

def filter_solar_irradiance(irradiance_data, min_irradiance=1200.0):
    """
    Filtrează terenurile în funcție de pragul minim de iradiere solară.
    """
    criteriu_iradiere = irradiance_data >= min_irradiance

    # Returnăm masca binară și matricea cu valorile brute
    return criteriu_iradiere, irradiance_data