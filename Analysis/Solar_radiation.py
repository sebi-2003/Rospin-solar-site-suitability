import numpy as np

def filter_solar_irradiance(irradiance_data, min_irradiance=1200.0):
    """
    Filtrează terenurile în funcție de pragul minim de iradiere solară anuală.

    Parametri:
    - irradiance_data: Matricea raster cu valorile de iradiere (kWh/m²/an).
    - min_irradiance: Pragul minim acceptat pentru eficiența unui parc solar (ex: 1200 kWh/m²/an).

    Returnează:
    - O mască booleană cu zonele care au o iradiere optimă.
    """
    # Selectăm doar zonele unde iradierea este mai mare sau egală cu pragul setat
    criteriu_iradiere = irradiance_data >= min_irradiance

    return criteriu_iradiere

def calculate_potential_radiation_index(slope_data, aspect_data):
    """
    [OPȚIONAL - Dacă nu aveți o hartă separată de iradiere]
    Calculează un indice geometric simplificat de expunere la soare
    pornind de la pantă și orientare (Aspect), util ca aproximare rapidă.
    """
    # Versanții orientați spre sud (180 grade) cu o pantă moderată (ex: 5 - 10 grade)
    # primesc unghiul optim de cădere a razelor solare.
    # Convertim aspectul într-un factor de penalizare față de sudul perfect (180°)
    deviation_from_south = np.abs(aspect_data - 180.0)
    # Factor de la 0 la 1, unde 1 este orientarea ideală spre sud
    aspect_factor = np.maximum(0.0, 1.0 - (deviation_from_south / 180.0))

    return aspect_factor