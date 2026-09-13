import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

def update_raster_with_forests():
    # Calea absolută către folderul curent și fișierul final
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Salvăm direct în folderul Backend
    backend_dir = os.path.join(os.path.dirname(current_dir), "Backend")
    os.makedirs(backend_dir, exist_ok=True)

    path_output = os.path.join(backend_dir, "rezultat_suitabilitate.tif")

    width, height = 1000, 1000

    # 1. Plecăm de la un teren considerat bun (valoare 85.0)
    data = np.full((height, width), 85.0, dtype=np.float32)

    # 2. Creăm o grilă de coordonate pentru a simula păduri/zone interzise (valoare 0.0)
    y, x = np.ogrid[:height, :width]

    # Lista zonelor împădurite (centru_y, centru_x, rază) - RAZE MICȘORATE
    forest_zones = [
        (300, 300, 20),  # Pădurea 1 (micșorată drastic de la 150)
        (700, 750, 30),  # Pădurea 2 (micșorată de la 180)
        (200, 800, 15)   # Pădurea 3 (micșorată de la 120)
    ]

    for cy_val, cx_val, r_val in forest_zones:
        mask_circle = (x - cx_val)**2 + (y - cy_val)**2 <= r_val**2
        data[mask_circle] = 0.0  # Valoarea 0 înseamnă excludere totală (pădure)

    transform = from_origin(23.55, 46.82, 0.0003, 0.0003)
    crs = CRS.from_epsg(4326)

    with rasterio.open(
            path_output,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(data, 1)

    print(f"Succes! Rasterul actualizat cu păduri mici a fost salvat în: {path_output}")

if __name__ == "__main__":
    update_raster_with_forests()