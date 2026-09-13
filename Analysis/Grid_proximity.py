import numpy as np
from scipy.ndimage import distance_transform_edt

def calculate_grid_proximity(grid_data, pixel_resolution=30.0):
    """
    Calculează distanța euclidiană (în metri) până la cea mai apropiată
    rețea electrică sau substație.
    """
    no_grid = (grid_data == 0)
    distance_in_pixels = distance_transform_edt(no_grid)
    distance_in_meters = distance_in_pixels * pixel_resolution
    return distance_in_meters

def filter_grid_proximity(grid_data, pixel_resolution=30.0, max_distance_meters=1000.0):
    """
    Filtrează terenurile fezabile în funcție de distanța maximă admisă față de rețea.
    Returnează masca binară și distanțele calculate.
    """
    distances = calculate_grid_proximity(grid_data, pixel_resolution)
    criteriu_retea = distances <= max_distance_meters
    return criteriu_retea, distances