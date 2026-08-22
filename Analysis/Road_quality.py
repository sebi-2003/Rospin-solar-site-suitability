import numpy as np
from scipy.ndimage import distance_transform_edt

def filter_heavy_transport_roads(road_classification_data, pixel_resolution=30.0, max_distance_meters=200.0):
    """
    Filtrează terenurile în funcție de accesul la drumuri principale (compatibile cu transport greu).
    Exclud drumurile forestiere înguste sau potecile.

    Parametri:
    - road_classification_data: Raster unde drumurile au coduri (ex: 1 = Drum Național/Județean, 2 = Drum Local, 3 = Forestier).
    - pixel_resolution: Rezoluția pixelului în metri.
    - max_distance_meters: Distanța maximă acceptată față de un drum principal.
    """
    print("\n[*] Se analizează accesul rutier pentru transport greu (trailere/camioane)...")

    # Presupunem că dorim doar drumurile principale (Cod 1 = Național/Județean)
    # Transformăm în 1 doar pixelii care reprezintă drumuri de mare tonaj
    heavy_roads = (road_classification_data == 1)

    if not np.any(heavy_roads):
        print("[ATENȚIE] Nu s-au găsit drumuri principale de categoria 1 pe hartă!")
        # Fallback de siguranță dacă e doar o hartă binară simplă
        heavy_roads = (road_classification_data > 0)

    # Calculăm distanța față de drumurile principale
    no_heavy_road = ~heavy_roads
    distance_pixels = distance_transform_edt(no_heavy_road)
    distance_meters = distance_pixels * pixel_resolution

    # Terenul este aprobat dacă se află la o distanță rezonabilă de un drum principal
    road_suitability = distance_meters <= max_distance_meters

    print(f"-> Analiză drumuri finalizată: Terenurile la max {max_distance_meters}m de drum greu sunt eligibile.")
    return road_suitability, distance_meters