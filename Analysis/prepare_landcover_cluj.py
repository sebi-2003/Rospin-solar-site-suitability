import os

import numpy as np
import rasterio

from rasterio.merge import merge
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds
)


# ============================================================
# CONFIGURARE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "Result"
)


# ============================================================
# CELE DOUĂ TILE-URI CLCPLUS CORECTE
# ============================================================

TILE_N26 = os.path.join(
    RESULT_DIR,
    "CLMS_CLCPLUS_RAS_S2023_R10m_E53N26_03035_V01_R00.tif"
)

TILE_N27 = os.path.join(
    RESULT_DIR,
    "CLMS_CLCPLUS_RAS_S2023_R10m_E53N27_03035_V01_R00.tif"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "Toate fisierele",
    "LandCover_Cluj_3844.tif"
)


# ============================================================
# ZONA CLUJULUI
# WGS84 / EPSG:4326
# ============================================================

WEST = 23.30
SOUTH = 46.50
EAST = 23.90
NORTH = 47.00


# ============================================================
# CRS FINAL
# ============================================================

TARGET_CRS = "EPSG:3844"


# ============================================================
# VERIFICARE FIȘIERE
# ============================================================

for path in [
    TILE_N26,
    TILE_N27
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nNu găsesc fișierul:\n{path}\n"
        )


print("\n" + "=" * 70)
print("PREGĂTIRE LAND COVER CLUJ")
print("=" * 70)

print("\nTile-uri:")

print(
    "1.",
    TILE_N26
)

print(
    "2.",
    TILE_N27
)


# ============================================================
# DESCHIDERE TILE-URI
# ============================================================

src1 = rasterio.open(
    TILE_N26
)

src2 = rasterio.open(
    TILE_N27
)

try:

    # --------------------------------------------------------
    # Verificăm CRS
    # --------------------------------------------------------

    print(
        "\nCRS tile 1:",
        src1.crs
    )

    print(
        "CRS tile 2:",
        src2.crs
    )

    print(
        "Rezoluție tile 1:",
        src1.res
    )

    print(
        "Rezoluție tile 2:",
        src2.res
    )

    print(
        "Dtype tile 1:",
        src1.dtypes[0]
    )

    print(
        "NoData tile 1:",
        src1.nodata
    )

    if src1.crs != src2.crs:

        raise ValueError(
            "Cele două tile-uri nu au același CRS."
        )

    source_crs = src1.crs

    if source_crs is None:

        raise ValueError(
            "Tile-urile nu au CRS definit."
        )


    # ========================================================
    # TRANSFORMĂM BOUNDING BOX CLUJ
    # DIN EPSG:4326 ÎN CRS-UL CLCPLUS
    # ========================================================

    cluj_bounds_source = transform_bounds(
        "EPSG:4326",
        source_crs,
        WEST,
        SOUTH,
        EAST,
        NORTH,
        densify_pts=21
    )

    print(
        "\nBounds Cluj în",
        source_crs,
        ":"
    )

    print(
        cluj_bounds_source
    )


    # ========================================================
    # MERGE + CLIP
    # ========================================================

    print(
        "\n[1/3] Unesc cele două tile-uri "
        "și tai zona Clujului..."
    )

    mosaic, mosaic_transform = merge(
        [
            src1,
            src2
        ],
        bounds=cluj_bounds_source,
        res=src1.res,
        nodata=src1.nodata
        if src1.nodata is not None
        else 255
    )


    if mosaic.shape[0] != 1:

        raise ValueError(
            "CLCplus trebuia să aibă "
            "o singură bandă."
        )


    source_data = mosaic[0]


    print(
        "Dimensiune după crop:",
        source_data.shape
    )

    print(
        "Transform:",
        mosaic_transform
    )


    # ========================================================
    # VERIFICARE VALORI CLCPLUS
    # ========================================================

    unique_values, counts = np.unique(
        source_data,
        return_counts=True
    )

    print("\nValori CLCplus găsite:")

    for value, count in zip(
            unique_values,
            counts
    ):

        print(
            f"  {value} -> "
            f"{count} pixeli"
        )


    # ========================================================
    # CONTROL IMPORTANT
    # ========================================================

    # CLCplus Backbone principal trebuie să aibă
    # clase mici 1..11 + eventual 253/254/255.

    valid_expected = set(
        range(1, 12)
    ) | {
                         0,
                         253,
                         254,
                         255
                     }

    actual_values = set(
        int(v)
        for v in unique_values
    )

    unexpected = (
            actual_values -
            valid_expected
    )

    if unexpected:

        raise ValueError(
            "\nATENȚIE!\n"
            "Rasterul conține valori care nu par "
            "CLCplus Backbone RAS:\n"
            f"{sorted(unexpected)}"
        )


    # ========================================================
    # BOUNDS ALE RASTERULUI CROPPED
    # ========================================================

    source_height = (
        source_data.shape[0]
    )

    source_width = (
        source_data.shape[1]
    )

    left = (
        mosaic_transform.c
    )

    top = (
        mosaic_transform.f
    )

    right = (
            left
            +
            mosaic_transform.a
            *
            source_width
    )

    bottom = (
            top
            +
            mosaic_transform.e
            *
            source_height
    )


    # ========================================================
    # CALCULĂM GRID-UL EPSG:3844
    # ========================================================

    print(
        "\n[2/3] Calculez reproiectarea "
        "în EPSG:3844..."
    )

    dst_transform, dst_width, dst_height = (
        calculate_default_transform(
            source_crs,
            TARGET_CRS,
            source_width,
            source_height,
            left,
            bottom,
            right,
            top,
            resolution=10
        )
    )


    print(
        "Dimensiune finală:",
        dst_width,
        "x",
        dst_height
    )

    print(
        "Rezoluție finală:",
        (
            abs(dst_transform.a),
            abs(dst_transform.e)
        )
    )


    # ========================================================
    # ARRAY DESTINAȚIE
    # ========================================================

    nodata_value = (
        src1.nodata
        if src1.nodata is not None
        else 255
    )

    destination = np.full(
        (
            dst_height,
            dst_width
        ),
        nodata_value,
        dtype=source_data.dtype
    )


    # ========================================================
    # REPROIECTARE
    # ========================================================

    print(
        "\n[3/3] Reproiectez cu "
        "NEAREST NEIGHBOUR..."
    )

    reproject(
        source=source_data,
        destination=destination,

        src_transform=mosaic_transform,
        src_crs=source_crs,
        src_nodata=nodata_value,

        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        dst_nodata=nodata_value,

        # FOARTE IMPORTANT:
        # Land Cover este categorial.
        resampling=Resampling.nearest
    )


    # ========================================================
    # CREĂM FOLDERUL OUTPUT
    # ========================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )


    # ========================================================
    # SALVARE
    # ========================================================

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": destination.dtype,
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "nodata": nodata_value,
        "compress": "lzw",
        "tiled": True
    }


    with rasterio.open(
            OUTPUT_PATH,
            "w",
            **profile
    ) as dst:

        dst.write(
            destination,
            1
        )


finally:

    src1.close()
    src2.close()


# ============================================================
# VERIFICARE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VERIFICARE LAND COVER FINAL")
print("=" * 70)

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
        "Dtype:",
        src.dtypes[0]
    )

    print(
        "NoData:",
        src.nodata
    )

    values, counts = np.unique(
        data,
        return_counts=True
    )

    print("\nVALORI FINALE:")

    for value, count in zip(
            values,
            counts
    ):

        print(
            f"  {value} -> "
            f"{count} pixeli"
        )


print("\n")
print("=" * 70)
print("LAND COVER CLUJ PREGĂTIT CU SUCCES")
print("=" * 70)

print(
    "\nOutput:\n",
    OUTPUT_PATH
)

print(
    "\nIMPORTANT:"
    "\nRasterul a fost reproiectat "
    "cu Resampling.nearest."
)