import os
import sys
import rasterio

def load_real_raster(file_path):
    """
    Încarcă un fișier raster real.
    Oprește execuția dacă fișierul nu este găsit în folder.
    """
    if not os.path.exists(file_path):
        print(f"[EROARE] Fișierul necesar nu a fost găsit: {file_path}")
        print("Asigurați-vă că harta este în folderul 'data/'.")
        sys.exit(1)

    with rasterio.open(file_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        # Extragem informații extra pentru verificare
        width = src.width
        height = src.height
        bounds = src.bounds

    return data, transform, crs, width, height, bounds

def test_loading_maps():
    print("=== MODUL: Încărcare și Verificare Hărți Reale ===")

    path_dem = "data/dem_teren.tif"
    path_roads = "data/drumuri.tif"

    print(f"\n[1/2] Se încearcă încărcarea DEM-ului: {path_dem}")
    dem_data, dem_transform, dem_crs, dem_w, dem_h, dem_bounds = load_real_raster(path_dem)
    print("  -> SUCCES! Date DEM încărcate.")
    print(f"  -> Dimensiuni: {dem_w}x{dem_h} pixeli")
    print(f"  -> CRS (Sistem de Coordonate): {dem_crs}")
    print(f"  -> Limite spațiale: {dem_bounds}")

    print(f"\n[2/2] Se încearcă încărcarea hărții de drumuri: {path_roads}")
    roads_data, roads_transform, roads_crs, roads_w, roads_h, roads_bounds = load_real_raster(path_roads)
    print("  -> SUCCES! Date drumuri încărcate.")
    print(f"  -> Dimensiuni: {roads_w}x{roads_h} pixeli")
    print(f"  -> CRS: {roads_crs}")

    # Verificare finală foarte importantă
    print("\n=== REZULTAT VERIFICARE ===")
    if (dem_w, dem_h) == (roads_w, roads_h):
        print("[OK] Hărțile au aceleași dimensiuni și pot fi suprapuse pentru analiză!")
    else:
        print("[ATENȚIE] Hărțile au dimensiuni diferite. Va trebui să le tăiați (clip) în QGIS la același perimetru!")

if __name__ == "__main__":
    test_loading_maps()