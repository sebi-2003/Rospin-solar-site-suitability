import os
import numpy as np
import rasterio
import fiona

from rasterio.features import rasterize
from rasterio.warp import (
    transform_bounds,
    transform_geom
)


# ============================================================
# CĂI PROIECT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

NATURA2000_PATH = os.path.join(
    PROJECT_ROOT,
    "AriiProtejate",
    "Natura2000_end2024.gpkg"
)

REFERENCE_RASTER = os.path.join(
    PROJECT_ROOT,
    "Toate fisierele",
    "LandCover_Cluj_3844.tif"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "Toate fisierele",
    "ProtectedAreas_Natura2000_Cluj_3844.tif"
)


# ============================================================
# VERIFICĂRI
# ============================================================

if not os.path.exists(NATURA2000_PATH):
    raise FileNotFoundError(
        f"Nu găsesc GeoPackage-ul:\n{NATURA2000_PATH}"
    )

if not os.path.exists(REFERENCE_RASTER):
    raise FileNotFoundError(
        f"Nu găsesc rasterul de referință:\n"
        f"{REFERENCE_RASTER}"
    )


print("\n" + "=" * 72)
print("PREGĂTIRE NATURA 2000 - CLUJ")
print("=" * 72)

print("\nGeoPackage:")
print(NATURA2000_PATH)

print("\nRaster referință:")
print(REFERENCE_RASTER)


# ============================================================
# CITIM GRIDUL DE REFERINȚĂ
# ============================================================

with rasterio.open(
        REFERENCE_RASTER
) as ref:

    target_crs = ref.crs
    target_transform = ref.transform
    target_width = ref.width
    target_height = ref.height
    target_bounds = ref.bounds

    print("\nCRS țintă:", target_crs)
    print("Rezoluție:", ref.res)
    print(
        "Dimensiune:",
        target_width,
        "x",
        target_height
    )
    print("Bounds:", target_bounds)


if target_crs is None:
    raise ValueError(
        "Rasterul de referință nu are CRS."
    )


# ============================================================
# STRATURILE DIN GEOPACKAGE
# ============================================================

layers = fiona.listlayers(
    NATURA2000_PATH
)

print("\nStraturi găsite în GPKG:")

for layer_name in layers:
    print(" -", layer_name)

if not layers:
    raise ValueError(
        "GeoPackage-ul nu conține niciun layer."
    )


# ============================================================
# COLECTĂM GEOMETRII NATURA 2000
# ============================================================

geometries = []

for layer_name in layers:

    print("\n" + "-" * 72)
    print("Layer:", layer_name)
    print("-" * 72)

    with fiona.open(
            NATURA2000_PATH,
            layer=layer_name
    ) as src:

        source_crs = src.crs

        print("CRS sursă:", source_crs)
        print(
            "Tip geometrie:",
            src.schema.get("geometry")
        )

        if not source_crs:
            print(
                "[SKIP] Layer fără CRS."
            )
            continue

        # ----------------------------------------------------
        # Bounds Cluj convertite în CRS-ul vectorului
        # ----------------------------------------------------

        bbox_source = transform_bounds(
            target_crs,
            source_crs,
            target_bounds.left,
            target_bounds.bottom,
            target_bounds.right,
            target_bounds.top,
            densify_pts=21
        )

        print(
            "BBox filtrare în CRS sursă:",
            bbox_source
        )

        found_in_layer = 0

        # ----------------------------------------------------
        # Citim doar obiectele care ating zona Cluj
        # ----------------------------------------------------

        try:
            features = src.filter(
                bbox=bbox_source
            )
        except Exception:
            # fallback dacă driverul nu suportă filtrarea
            features = src

        for feature in features:

            geometry = feature.get(
                "geometry"
            )

            if geometry is None:
                continue

            geom_type = geometry.get(
                "type",
                ""
            )

            # Ne interesează doar suprafețele
            if geom_type not in (
                    "Polygon",
                    "MultiPolygon"
            ):
                continue

            try:

                geometry_3844 = transform_geom(
                    source_crs,
                    target_crs,
                    geometry,
                    precision=3
                )

                geometries.append(
                    geometry_3844
                )

                found_in_layer += 1

            except Exception as exc:

                print(
                    "[WARNING] Geometrie ignorată:",
                    exc
                )

        print(
            "Geometrii găsite în zona Cluj:",
            found_in_layer
        )


# ============================================================
# VERIFICARE
# ============================================================

print("\n" + "=" * 72)

print(
    "TOTAL GEOMETRII NATURA 2000 "
    "FOLOSITE:",
    len(geometries)
)

print("=" * 72)


if len(geometries) == 0:

    raise ValueError(
        "\nNu am găsit nicio geometrie Natura 2000 "
        "în zona rasterului Cluj.\n"
        "Verifică layer-ele și CRS-urile."
    )


# ============================================================
# RASTERIZARE
# ============================================================

print(
    "\nRasterizez ariile Natura 2000..."
)

protected_mask = rasterize(
    (
        (geometry, 1)
        for geometry in geometries
    ),
    out_shape=(
        target_height,
        target_width
    ),
    transform=target_transform,
    fill=0,
    dtype="uint8",
    all_touched=False
)


# ============================================================
# STATISTICI
# ============================================================

protected_pixels = int(
    np.count_nonzero(
        protected_mask == 1
    )
)

total_pixels = (
    protected_mask.size
)

protected_percentage = (
        protected_pixels
        /
        total_pixels
        *
        100.0
)


print("\nSTATISTICI:")

print(
    "Pixeli protejați:",
    protected_pixels
)

print(
    "Pixeli total:",
    total_pixels
)

print(
    f"Procent raster protejat: "
    f"{protected_percentage:.4f}%"
)


# ============================================================
# SALVARE
# ============================================================

profile = {
    "driver": "GTiff",
    "height": target_height,
    "width": target_width,
    "count": 1,
    "dtype": "uint8",
    "crs": target_crs,
    "transform": target_transform,

    # 0 = în afara Natura 2000
    # 1 = Natura 2000
    "nodata": None,

    "compress": "lzw",
    "tiled": True
}


with rasterio.open(
        OUTPUT_PATH,
        "w",
        **profile
) as dst:

    dst.write(
        protected_mask,
        1
    )


# ============================================================
# VERIFICARE FINALĂ
# ============================================================

print("\n" + "=" * 72)
print("VERIFICARE RASTER NATURA 2000")
print("=" * 72)

with rasterio.open(
        OUTPUT_PATH
) as src:

    data = src.read(1)

    print(
        "Fișier:",
        OUTPUT_PATH
    )

    print(
        "CRS:",
        src.crs
    )

    print(
        "Rezoluție:",
        src.res
    )

    print(
        "Dimensiune:",
        src.width,
        "x",
        src.height
    )

    print(
        "Valori:",
        np.unique(data)
    )


print("\n" + "=" * 72)
print("NATURA 2000 CLUJ PREGĂTIT CU SUCCES")
print("=" * 72)

print(
    "\nOutput:\n",
    OUTPUT_PATH
)

print(
    "\nLegendă:"
    "\n0 = în afara Natura 2000"
    "\n1 = sit Natura 2000"
)