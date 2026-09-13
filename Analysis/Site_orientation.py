import numpy as np

def calculate_slope(dem, resolution=30.0):
    """
    Calculează înclinația pantei (Slope) în grade, esențială pentru analiză și frontend.
    """
    dy, dx = np.gradient(dem, resolution, resolution)
    slope_percent = np.hypot(dx, dy)
    slope_degrees = np.arctan(slope_percent) * (180.0 / np.pi)
    return slope_degrees

def calculate_aspect(dem, resolution=30.0):
    """
    Calculează orientarea versanților (Aspect) în grade (0 - 360) față de Nord.
    """
    dy, dx = np.gradient(dem, resolution, resolution)
    aspect = np.arctan2(-dy, dx) * (180.0 / np.pi)
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    return aspect

def filter_optimal_aspect(dem_data, resolution=30.0):
    """
    Filtrează orientările favorabile pentru parcuri solare (Sector sudic: 135° - 225°).
    """
    aspect_data = calculate_aspect(dem_data, resolution)
    criteriu_sud = (aspect_data >= 135) & (aspect_data <= 225)
    return criteriu_sud, aspect_data