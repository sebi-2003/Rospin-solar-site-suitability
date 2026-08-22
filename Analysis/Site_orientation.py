import numpy as np

def calculate_aspect(dem, resolution=1.0):
    """
    Calculează orientarea versanților (Aspect) în grade (0 - 360) față de Nord.
    - 0° / 360° = Nord
    - 90° = Est
    - 180° = Sud
    - 270° = Vest
    """
    # Calculăm gradientul pe cele două direcții (nord-sud și est-vest)
    dy, dx = np.gradient(dem, resolution, resolution)

    # Calculăm aspectul în radiani și îl convertim în grade
    aspect = np.arctan2(-dy, dx) * (180.0 / np.pi)

    # Corectăm valorile negative pentru a fi în intervalul [0, 360]
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    return aspect

def filter_optimal_aspect(aspect_data):
    """
    Filtrează orientările favorabile pentru parcuri solare în emisfera nordică.
    Considerăm optim sectorul sudic: între 135° (Sud-Est) și 225° (Sud-Vest),
    dar acceptăm și o marjă mai largă (de la Est la Vest: 90° - 270°).
    """
    # Variantă strictă (doar versanți orientați spre Sud / SE / SW):
    criteriu_sud = (aspect_data >= 135) & (aspect_data <= 225)

    # Sau variantă mai relaxată (orice versant care bate spre soare: Est, Sud, Vest):
    # criteriu_soare = (aspect_data >= 90) & (aspect_data <= 270)

    return criteriu_sud