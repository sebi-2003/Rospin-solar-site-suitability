import rasterio
from rasterio.warp import transform_bounds

# Asigură-te că calea către fișier este corectă (dacă e în folderul principal, folosește "../rezultat_suitabilitate.tif")
with rasterio.open("D:/Rospin-solar-site-suitability/Backend/rezultat_suitabilitate.tif") as src:
    # Dacă rasterul nu are CRS, îi setăm temporar EPSG:31700 (sau cel folosit de hărțile tale)
    raster_crs = src.crs if src.crs is not None else 'EPSG:31700'

    left, bottom, right, top = transform_bounds(raster_crs, 'EPSG:4326', *src.bounds)

    center_lat = (bottom + top) / 2
    center_lon = (left + right) / 2

    print("==========================================")
    print("  COORDONATELE GEOGRAFICE ALE HĂRȚII:")
    print("==========================================")
    print(f"Colțul Sud-Vest (Min): Lat {bottom:.5f}, Lon {left:.5f}")
    print(f"Colțul Nord-Est (Max): Lat {top:.5f}, Lon {right:.5f}")
    print(f"CENTRUL HĂRȚII (Unde să te uiți): {center_lat:.5f}, {center_lon:.5f}")
    print("==========================================")