import os
import rasterio
import numpy as np

# Importăm modulele din folderul Analysis
from Analysis.Exclusion_masks import filter_exclusion_zones
from Analysis.Grid_proximity import filter_grid_proximity
from Analysis.Site_orientation import calculate_aspect, filter_optimal_aspect
from Analysis.Solar_radiation import filter_solar_irradiance
from Analysis.Road_quality import filter_heavy_transport_roads
from Analysis.Buffer_zones import check_buffer_zones
from Analysis.Mcda_analysis import compute_mcda_score
from Analysis.Minimum_area_filter import check_one_hectare_limit

def run_full_pipeline():
    print("==================================================")
    print("  PORNIRE PIPELINE COMPLET DE ANALIZĂ GIS - SOLAR ")
    print("==================================================")

    # Căile către hărțile raster brute din folderul data/
    dem_path = "data/dem_teren.tif"
    roads_path = "data/drumuri.tif"
    grid_path = "data/retea_electrica.tif"
    irradiance_path = "data/iradiere.tif"
    land_cover_path = "data/land_cover.tif"

    # Verificare de siguranță pentru folderul data
    if not os.path.exists(dem_path):
        print(f"\n[EROARE CRITICă] Nu s-a găsit fișierul DEM de bază: {dem_path}")
        print("Asigură-te că ai pus hărțile reale în folderul 'data/' înainte de rulare.")
        return

    print("\n[1/8] Se încarcă hărțile raster brute...")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        profile = src.profile

    with rasterio.open(roads_path) as src:
        roads_data = src.read(1)

    with rasterio.open(grid_path) as src:
        grid_data = src.read(1)

    with rasterio.open(irradiance_path) as src:
        irradiance_data = src.read(1)

    with rasterio.open(land_cover_path) as src:
        land_cover_data = src.read(1)
    print(" -> Toate hărțile au fost încărcate cu succes.")

    # [2/8] Pasul de excludere legală și de mediu
    mask_legal = filter_exclusion_zones(land_cover_data)

    # [3/8] Proximitatea față de rețeaua electrică (max 1000m)
    mask_grid, dist_grid = filter_grid_proximity(grid_data, pixel_resolution=abs(transform[0]), max_distance_meters=1000.0)

    # [4/8] Orientarea versanților (Aspect spre soare)
    aspect = calculate_aspect(dem_data)
    mask_aspect = filter_optimal_aspect(aspect)

    # [5/8] Iradierea solară minimă (ex: min 1200 kWh/m²/an)
    mask_irradiance = filter_solar_irradiance(irradiance_data, min_irradiance=1200.0)

    # [6/8] Accesul la drumuri pentru transport greu (max 200m)
    mask_roads, dist_roads = filter_heavy_transport_roads(roads_data, pixel_resolution=abs(transform[0]), max_distance_meters=200.0)

    # [7/8] Verificarea zonelor tampon (Buffer de 50m)
    mask_buffers = check_buffer_zones(roads_data, pixel_resolution=abs(transform[0]), buffer_distance_meters=50.0)

    print("\n[8/8] Se aplică intersecția globală a tuturor filtrelor și calculul MCDA...")
    # Intersecția booleană strictă a tuturor condițiilor de inginerie
    final_boolean_mask = np.logical_and(
        np.logical_and(mask_legal, mask_grid),
        np.logical_and(
            np.logical_and(mask_aspect, mask_irradiance),
            np.logical_and(mask_roads, mask_buffers)
        )
    )

    # Calculăm panta aproximativă din DEM pentru scorul MCDA
    dy, dx = np.gradient(dem_data, abs(transform[0]))
    slope_data = np.arctan(np.sqrt(dx**2 + dy**2)) * (180.0 / np.pi)

    # Generăm harta de tip Heatmap (0 - 100) pe baza ponderilor stabilite
    mcda_scores = compute_mcda_score(slope_data, irradiance_data, dist_grid, dist_roads)

    # Păstrăm scorurile doar în zonele care au trecut de toate filtrele de suitabilitate
    final_scored_map = np.where(final_boolean_mask, mcda_scores, 0)

    # Verificarea finală a suprafeței în hectare
    check_one_hectare_limit(final_boolean_mask, transform)

    # Salvăm rezultatul final într-un nou GeoTIFF pe care îl puteți deschide direct în QGIS
    output_filename = "rezultat_final_suitabilitate.tif"
    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_filename, "w", **profile) as dst:
        dst.write(final_scored_map.astype(rasterio.float32), 1)

    print("==================================================")
    print(f"  ANALIZĂ FINALIZATĂ CU SUCCES!")
    print(f"  Fișierul generat pentru QGIS: '{output_filename}'")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()