import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import label, distance_transform_edt

def load_real_raster(file_path):
    """
    Funcție pentru citirea unui raster real.
    Dacă fișierul lipsește, generează automat o matrice simulată pentru a preveni erorile.
    """
    if os.path.exists(file_path):
        with rasterio.open(file_path) as src:
            return src.read(1), src.transform, src.crs
    else:
        print(f"[ATENȚIE] Fișierul nu a fost găsit: {file_path}. Se folosesc date simulate.")
        # Generăm o matrice de 100x100 de tip float pentru a simula un DEM
        dummy_data = np.random.uniform(200, 500, (100, 100)).astype(np.float32)
        dummy_transform = from_origin(0, 100, 1, 1) # xmin, ymax, xsize, ysize
        return dummy_data, dummy_transform, None

def calculate_slope(dem, resolution=1.0):
    """ Calculează panta (slope) în grade pornind de la Modelul Digital de Elevație (DEM). """
    dy, dx = np.gradient(dem, resolution, resolution)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180.0 / np.pi)
    return slope

def real_site_suitability_analysis():
    print("=== PROIECT GIS: Analiză Multi-Criterială pentru Parcuri Solare ===")

    # 1. Definirea căilor (ajustează-le ulterior dacă ai datele reale)
    path_dem = "data/dem_teren.tif"
    path_roads = "data/drumuri.tif"
    path_output = "rezultat_suitabilitate.tif"

    # 2. Încărcarea datelor
    print("[1/5] Încărcare Model Digital al Elevației (DEM)...")
    dem_data, transform, crs = load_real_raster(path_dem)

    print("[2/5] Încărcare hartă infrastructură/drumuri...")
    roads_data, _, _ = load_real_raster(path_roads)

    # Dacă folosim date simulate, transformăm harta de drumuri într-o matrice binară (1=drum, 0=teren)
    if not os.path.exists(path_roads):
        roads_data = np.random.choice([0, 1], size=dem_data.shape, p=[0.98, 0.02])

    # 3. Calcularea Criteriilor pentru Panouri Solare
    print("[3/5] Calculare pantă (Slope)...")
    # Criteriul 1: Teren relativ plat (pantă sub 15 grade)
    slope_data = calculate_slope(dem_data)
    criteriu_panta = slope_data < 15.0

    print("[4/5] Calculare proximitate față de drumuri...")
    # Criteriul 2: Să fie aproape de un drum pentru acces facil
    # distance_transform_edt calculează distanța față de valorile de 0, deci inversăm matricea
    lipsa_drum = (roads_data == 0)
    distanta_drumuri = distance_transform_edt(lipsa_drum)
    criteriu_distanta = distanta_drumuri < 25  # distanța acceptată

    # 4. Intersectarea Criteriilor (Boolean Logic)
    print("[5/5] Generare hartă finală de suitabilitate...")
    suitabilitate_finala = np.logical_and(criteriu_panta, criteriu_distanta).astype(np.uint8)

    # Identificăm și numărăm parcelele continue (clusterele valide)
    labeled_array, num_features = label(suitabilitate_finala)
    print(f"-> Au fost identificate {num_features} parcele potențiale care respectă toate criteriile!")

    # 5. Salvarea rezultatului în format GeoTIFF
    with rasterio.open(
            path_output,
            'w',
            driver='GTiff',
            height=suitabilitate_finala.shape[0],
            width=suitabilitate_finala.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(suitabilitate_finala, 1)

    print(f"=== SUCCES! Harta finală a fost salvată în folderul curent: {path_output} ===")

if __name__ == "__main__":
    real_site_suitability_analysis()