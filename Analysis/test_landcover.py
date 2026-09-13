import os
import rasterio
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BASE_DIR
)

LANDCOVER_PATH = os.path.join(
    PROJECT_ROOT,
    "Toate fisierele",
    "LandCover_corect_taiat.tif"
)


print("=" * 70)
print("VERIFICARE LAND COVER")
print("=" * 70)

print("Fișier:")
print(LANDCOVER_PATH)

print()

print(
    "Există:",
    os.path.exists(LANDCOVER_PATH)
)

if not os.path.exists(LANDCOVER_PATH):
    raise FileNotFoundError(
        LANDCOVER_PATH
    )


with rasterio.open(LANDCOVER_PATH) as src:

    data = src.read(1)

    print()
    print("CRS:", src.crs)
    print("Dimensiune:", src.width, "x", src.height)
    print("Rezoluție:", src.res)
    print("Dtype:", src.dtypes[0])
    print("NoData:", src.nodata)
    print("Bounds:", src.bounds)

    print()
    print("=" * 70)
    print("VALORI UNICE")
    print("=" * 70)

    values, counts = np.unique(
        data,
        return_counts=True
    )

    for value, count in zip(
            values[:100],
            counts[:100]
    ):
        print(
            f"{value} -> {count} pixeli"
        )

    print()
    print(
        "Număr total de valori unice:",
        len(values)
    )

    finite = data[
        np.isfinite(data)
    ]

    if finite.size > 0:

        print(
            "MIN:",
            np.min(finite)
        )

        print(
            "MAX:",
            np.max(finite)
        )

        print(
            "MEAN:",
            np.mean(finite)
        )