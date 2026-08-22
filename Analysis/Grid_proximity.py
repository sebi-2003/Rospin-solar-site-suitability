import numpy as np
from scipy.ndimage import distance_transform_edt

def calculate_grid_proximity(grid_data, pixel_resolution=30.0):
    """
    Calculează distanța euclidiană (în metri) până la cea mai apropiată
    rețea electrică sau substație.

    Parametri:
    - grid_data: Matrice binară unde 1 = rețea electrică/stâlp/substație, 0 = teren liber.
    - pixel_resolution: Dimensiunea unui pixel în metri (preluat din georeferențierea rasterului, ex: 30m).

    Returnează:
    - o matrice cu distanțele reale în metri pentru fiecare punct de pe hartă.
    """
    # Inversăm masca: transformăm în 0 zonele unde există rețea și 1 unde nu există
    no_grid = (grid_data == 0)

    # Calculăm distanța în număr de pixeli față de cel mai apropiat element de rețea (valoarea 1)
    distance_in_pixels = distance_transform_edt(no_grid)

    # Convertim din pixeli în metri înmulțind cu rezoluția spațială
    distance_in_meters = distance_in_pixels * pixel_resolution

    return distance_in_meters

def filter_grid_proximity(grid_data, pixel_resolution=30.0, max_distance_meters=1000.0):
    """
    Filtrează terenurile fezabile din punct de vedere economic în funcție de
    distanța maximă admisă față de rețea (ex: 1000 metri / 1 km).
    """
    distances = calculate_grid_proximity(grid_data, pixel_resolution)

    # Terenul este eligibil dacă se află la o distanță mai mică sau egală cu pragul maxim
    criteriu_retea = distances <= max_distance_meters

    return criteriu_retea, distances